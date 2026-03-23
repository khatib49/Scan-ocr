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

import warnings
from typing import Dict, Any, Optional

import cv2
import numpy as np
import urllib.request

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# URL → BYTES HELPER
# ─────────────────────────────────────────────

def _fetch_image_bytes(url: str, timeout: int = 10) -> bytes:
    """
    Download image from a URL and return raw bytes.
    Raises ValueError if download fails.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ScanInvoiceAPI/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        raise ValueError(f"Failed to fetch image from URL: {e}") from e


# ─────────────────────────────────────────────
# INDIVIDUAL DETECTORS
# ─────────────────────────────────────────────

def _detect_vignetting(gray: np.ndarray):
    """Returns (score 0-100, ratio). High = strong vignetting = screen photo."""
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
    return (100 if ratio > 2.0 else (60 if ratio > 1.5 else 10)), ratio


def _detect_bezel(gray: np.ndarray):
    """Returns (score 0-100, max_ratio). High = dark borders = phone bezel visible."""
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


def _detect_refresh_banding(gray: np.ndarray):
    """Returns (score 0-100, ratio). High = horizontal banding = screen refresh artifact."""
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


def _detect_blur_inconsistency(gray: np.ndarray, grid: int = 6):
    """Returns (score 0-100, ratio). High = uneven focus = screen in frame."""
    h, w = gray.shape
    gh, gw = h // grid, w // grid
    sharp_map = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            patch = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            sharp_map[i, j] = cv2.Laplacian(patch, cv2.CV_64F).var()
    ratio = sharp_map.std() / (sharp_map.mean() + 1e-6)
    return (100 if ratio > 1.2 else (60 if ratio > 0.8 else 10)), ratio


def _detect_fft_grid(gray: np.ndarray):
    """Returns (score 0-100, peaks). High = periodic pixel grid = photographed screen."""
    f = np.fft.fft2(gray.astype(np.float32))
    fs = np.fft.fftshift(f)
    mag = np.log(np.abs(fs) + 1)
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    mag[cy - 15:cy + 15, cx - 15:cx + 15] = 0
    threshold = np.percentile(mag, 99.5)
    peaks = int((mag > threshold).sum())
    return (100 if peaks > 500 else (80 if peaks > 100 else (40 if peaks > 50 else 5))), peaks


def _detect_rgb_correlation(rgb: np.ndarray):
    """Returns (score 0-100, avg_corr). High = channels too similar = screen emission."""
    r = rgb[:, :, 0].astype(np.float32).flatten()
    g = rgb[:, :, 1].astype(np.float32).flatten()
    b = rgb[:, :, 2].astype(np.float32).flatten()
    avg_corr = (np.corrcoef(r, g)[0, 1] + np.corrcoef(r, b)[0, 1] + np.corrcoef(g, b)[0, 1]) / 3
    return (100 if avg_corr > 0.97 else (60 if avg_corr > 0.90 else 10)), avg_corr


def _detect_glare(rgb: np.ndarray):
    """Returns (score 0-100, glare_ratio). High = specular highlights = glass screen."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    near_white = (r > 230) & (g > 230) & (b > 230)
    glare_ratio = near_white.sum() / (rgb.shape[0] * rgb.shape[1]) * 100
    return (100 if glare_ratio > 3.0 else (60 if glare_ratio > 1.0 else 10)), glare_ratio


def _detect_color_temperature(rgb: np.ndarray):
    """Returns (score 0-100, br_ratio). High = blue-shifted = cool screen light."""
    r_m = rgb[:, :, 0].astype(np.float32).mean()
    b_m = rgb[:, :, 2].astype(np.float32).mean()
    br_ratio = b_m / (r_m + 1e-6)
    score = 80 if (0.95 < br_ratio < 1.10) else (40 if br_ratio > 0.90 else 5)
    return score, br_ratio


# ─────────────────────────────────────────────
# WEIGHTS
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
THRESHOLD_MANUAL_REVIEW = 50   # 50-69  → flag only, processing continues
THRESHOLD_SOFT_FLAG     = 25   # 25-49  → soft flag


# ─────────────────────────────────────────────
# CORE DETECTION LOGIC (works on decoded image)
# ─────────────────────────────────────────────

def _run_detection(image_bytes: bytes) -> Dict[str, Any]:
    """
    Core detection — takes raw image bytes, returns full result dict.
    """
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2 could not decode image")
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return {
            "is_screen_photo": False,
            "confidence":      "clean",
            "score":           0.0,
            "action":          "approve",
            "details":         f"Image decode failed: {e}",
            "detectors":       {}
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
        action     = "auto_reject"
        details    = "High confidence screen photo — receipt photographed from a phone screen."
    elif final_score >= THRESHOLD_MANUAL_REVIEW:
        confidence = "medium"
        action     = "manual_review"
        details    = "Suspicious screen photo indicators detected — flagged for review."
    elif final_score >= THRESHOLD_SOFT_FLAG:
        confidence = "low"
        action     = "soft_flag"
        details    = "Some suspicious indicators — proceed with caution."
    else:
        confidence = "clean"
        action     = "approve"
        details    = "No significant screen-photo indicators detected."

    return {
        "is_screen_photo": final_score >= THRESHOLD_MANUAL_REVIEW,
        "confidence":      confidence,
        "score":           round(final_score, 2),
        "action":          action,
        "details":         details,
        "detectors": {
            k: {"score": scores[k], "raw": round(float(raws[k]), 4)}
            for k in scores
        }
    }


# ─────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ─────────────────────────────────────────────

def detect_screen_photo(image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    """
    Detect screen photo from raw image bytes.
    Used in /analyze endpoint where bytes are already in memory.
    """
    return _run_detection(image_bytes)


def detect_screen_photo_from_url(image_url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Detect screen photo from an image URL (e.g. Azure Blob SAS URL).
    Downloads the image first, then runs detection.

    Usage:
        result = detect_screen_photo_from_url(
            "https://invoicescannerstorage.blob.core.windows.net/invoicefiles/receipt.jpg?..."
        )

    Returns same structure as detect_screen_photo().
    """
    try:
        image_bytes = _fetch_image_bytes(image_url, timeout=timeout)
    except ValueError as e:
        return {
            "is_screen_photo": False,
            "confidence":      "clean",
            "score":           0.0,
            "action":          "approve",
            "details":         str(e),
            "detectors":       {}
        }

    return _run_detection(image_bytes)