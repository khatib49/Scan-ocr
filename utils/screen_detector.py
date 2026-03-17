# utils/screen_detector.py
"""
Screen-Photographed Receipt Detector
Detects receipts photographed from a phone/screen instead of directly.

Detectors:
  1. Vignetting          - Camera darkens corners when shooting a screen
  2. Bezel Detection     - Dark phone frame visible around screen content
  3. Refresh Banding     - Screen Hz creates horizontal brightness waves
  4. Blur Inconsistency  - Screen sharp, real-world background blurry
  5. FFT Pixel Grid      - Screen emits periodic pixel frequency
  6. RGB Correlation     - Phone screen emits perfectly uniform RGB light
  7. Glare/Reflection    - Specular highlights on glass screen surface
  8. Color Temperature   - Screen light is blue-shifted vs warm paper
"""

import io
import warnings
from typing import Dict, Any

import cv2
import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# INDIVIDUAL DETECTORS
# ─────────────────────────────────────────────

def _detect_vignetting(gray: np.ndarray) -> float:
    """Returns score 0-100. High = strong vignetting = screen photo."""
    h, w = gray.shape
    thick = max(30, h // 15)
    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4].mean()
    edges = np.mean([
        gray[0:thick, :].mean(),
        gray[h - thick:h, :].mean(),
        gray[:, 0:thick].mean(),
        gray[:, w - thick:w].mean(),
    ])
    ratio = center / (edges + 1e-6)
    return 100 if ratio > 2.0 else (60 if ratio > 1.5 else 10), ratio


def _detect_bezel(gray: np.ndarray) -> float:
    """Returns score 0-100. High = dark borders = phone bezel visible."""
    h, w = gray.shape
    thick = max(30, w // 12)
    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4].mean()
    ratios = [
        center / (gray[:, 0:thick].mean() + 1e-6),
        center / (gray[:, w - thick:w].mean() + 1e-6),
        center / (gray[0:thick, :].mean() + 1e-6),
        center / (gray[h - thick:h, :].mean() + 1e-6),
    ]
    max_r = max(ratios)
    return (100 if max_r > 5.0 else (80 if max_r > 2.0 else (40 if max_r > 1.5 else 5))), max_r


def _detect_refresh_banding(gray: np.ndarray) -> float:
    """Returns score 0-100. High = horizontal banding = screen refresh artifact."""
    row_means = gray.mean(axis=1).astype(np.float32)
    trend = np.convolve(row_means, np.ones(50) / 50, mode="same")
    detrended = row_means - trend
    fft_rows = np.abs(np.fft.fft(detrended))
    fft_rows[0] = 0
    half = len(fft_rows) // 2
    dominant = fft_rows[1:half].max()
    avg = fft_rows[1:half].mean()
    ratio = dominant / (avg + 1e-6)
    return (100 if ratio > 20 else (60 if ratio > 10 else 5)), ratio


def _detect_blur_inconsistency(gray: np.ndarray, grid: int = 6) -> float:
    """Returns score 0-100. High = uneven focus = screen in frame, background blurry."""
    h, w = gray.shape
    gh, gw = h // grid, w // grid
    sharp_map = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            patch = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            sharp_map[i, j] = cv2.Laplacian(patch, cv2.CV_64F).var()
    ratio = sharp_map.std() / (sharp_map.mean() + 1e-6)
    return (100 if ratio > 1.2 else (60 if ratio > 0.8 else 10)), ratio


def _detect_fft_grid(gray: np.ndarray) -> float:
    """Returns score 0-100. High = periodic pixel grid = photographed screen."""
    f = np.fft.fft2(gray.astype(np.float32))
    fs = np.fft.fftshift(f)
    mag = np.log(np.abs(fs) + 1)
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    mag[cy - 15:cy + 15, cx - 15:cx + 15] = 0
    threshold = np.percentile(mag, 99.5)
    peaks = int((mag > threshold).sum())
    return (100 if peaks > 500 else (80 if peaks > 100 else (40 if peaks > 50 else 5))), peaks


def _detect_rgb_correlation(rgb: np.ndarray) -> float:
    """Returns score 0-100. High = channels too similar = screen uniform emission."""
    r = rgb[:, :, 0].astype(np.float32).flatten()
    g = rgb[:, :, 1].astype(np.float32).flatten()
    b = rgb[:, :, 2].astype(np.float32).flatten()
    avg_corr = (np.corrcoef(r, g)[0, 1] + np.corrcoef(r, b)[0, 1] + np.corrcoef(g, b)[0, 1]) / 3
    return (100 if avg_corr > 0.97 else (60 if avg_corr > 0.90 else 10)), avg_corr


def _detect_glare(rgb: np.ndarray) -> float:
    """Returns score 0-100. High = specular highlights = glass screen surface."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    near_white = (r > 230) & (g > 230) & (b > 230)
    glare_ratio = near_white.sum() / (rgb.shape[0] * rgb.shape[1]) * 100
    return (100 if glare_ratio > 3.0 else (60 if glare_ratio > 1.0 else 10)), glare_ratio


def _detect_color_temperature(rgb: np.ndarray) -> float:
    """Returns score 0-100. High = blue-shifted = cool screen light."""
    r_m = rgb[:, :, 0].astype(np.float32).mean()
    b_m = rgb[:, :, 2].astype(np.float32).mean()
    br_ratio = b_m / (r_m + 1e-6)
    score = 80 if (0.95 < br_ratio < 1.10) else (40 if br_ratio > 0.90 else 5)
    return score, br_ratio


# ─────────────────────────────────────────────
# WEIGHTS  (must sum to 17)
# ─────────────────────────────────────────────
_WEIGHTS = {
    "vignetting":         3,
    "bezel":              3,
    "banding":            2,
    "blur":               2,
    "fft":                3,
    "rgb_correlation":    2,
    "glare":              1,
    "color_temperature":  1,
}

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
THRESHOLD_AUTO_REJECT   = 70   # >= 70  → auto reject
THRESHOLD_MANUAL_REVIEW = 50   # 51-69  → manual review
THRESHOLD_SOFT_FLAG     = 25   # 25-50  → soft flag


# ─────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────

def detect_screen_photo(image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    """
    Analyze image bytes and return screen-photo detection result.

    Returns:
    {
        "is_screen_photo": bool,
        "confidence": "high" | "medium" | "low" | "clean",
        "score": float (0-100),
        "action": "auto_reject" | "manual_review" | "soft_flag" | "approve",
        "details": str,
        "detectors": {
            "vignetting":        {"score": int, "raw": float},
            "bezel":             {"score": int, "raw": float},
            "banding":           {"score": int, "raw": float},
            "blur":              {"score": int, "raw": float},
            "fft":               {"score": int, "raw": float},
            "rgb_correlation":   {"score": int, "raw": float},
            "glare":             {"score": int, "raw": float},
            "color_temperature": {"score": int, "raw": float},
        }
    }
    """
    # Decode image
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2 could not decode image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return {
            "is_screen_photo": False,
            "confidence": "clean",
            "score": 0.0,
            "action": "approve",
            "details": f"Image decode failed: {e}",
            "detectors": {}
        }

    # Run all detectors
    vig_s,  vig_r  = _detect_vignetting(img_gray)
    bez_s,  bez_r  = _detect_bezel(img_gray)
    ban_s,  ban_r  = _detect_refresh_banding(img_gray)
    blur_s, blur_r = _detect_blur_inconsistency(img_gray)
    fft_s,  fft_r  = _detect_fft_grid(img_gray)
    rgb_s,  rgb_r  = _detect_rgb_correlation(img_rgb)
    gla_s,  gla_r  = _detect_glare(img_rgb)
    tmp_s,  tmp_r  = _detect_color_temperature(img_rgb)

    scores = {
        "vignetting":        vig_s,
        "bezel":             bez_s,
        "banding":           ban_s,
        "blur":              blur_s,
        "fft":               fft_s,
        "rgb_correlation":   rgb_s,
        "glare":             gla_s,
        "color_temperature": tmp_s,
    }
    raws = {
        "vignetting":        vig_r,
        "bezel":             bez_r,
        "banding":           ban_r,
        "blur":              blur_r,
        "fft":               fft_r,
        "rgb_correlation":   rgb_r,
        "glare":             gla_r,
        "color_temperature": tmp_r,
    }

    # Weighted final score
    total_w  = sum(scores[k] * _WEIGHTS[k] for k in scores)
    total_wt = sum(_WEIGHTS.values())
    final_score = total_w / total_wt

    # Verdict
    if final_score >= THRESHOLD_AUTO_REJECT:
        confidence = "high"
        action = "auto_reject"
        details = "High confidence screen photo — receipt photographed from a phone screen."
    elif final_score >= THRESHOLD_MANUAL_REVIEW:
        confidence = "medium"
        action = "manual_review"
        details = "Suspicious screen photo indicators detected — flagged for review."
    elif final_score >= THRESHOLD_SOFT_FLAG:
        confidence = "low"
        action = "soft_flag"
        details = "Some suspicious indicators — proceed with caution."
    else:
        confidence = "clean"
        action = "approve"
        details = "No significant screen-photo indicators detected."

    is_screen = final_score >= THRESHOLD_MANUAL_REVIEW  # medium or high = flagged

    return {
        "is_screen_photo": is_screen,
        "confidence": confidence,
        "score": round(final_score, 2),
        "action": action,
        "details": details,
        "detectors": {
            k: {"score": scores[k], "raw": round(float(raws[k]), 4)}
            for k in scores
        }
    }