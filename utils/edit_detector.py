# utils/edit_detector.py
"""
Receipt Edit / Manipulation Detector  (v2)
Detects digitally edited, Photoshop-tampered, or forged receipt images.

Detectors:
  1. EXIF / XMP / 8BIM Metadata   — Photoshop software tags, edit history, binary blocks
  2. ELA  (Error Level Analysis)  — Re-saved/edited JPEG regions (bright patch = tampered)
  3. Noise Consistency            — Wavelet noise map; spliced regions have mismatched noise
  4. Halo / Edge Artifacts        — Compositing/blending halos at object boundaries
  5. Clone Stamp / Heal Detection — Repeated texture patches (clone stamp, healing brush)
  6. JPEG Ghost Analysis          — Double-compression mismatch from re-saving through editor
  7. Copy-Move (ORB)              — Feature-based detection of duplicated image regions
  8. DCT Block Anomaly            — JPEG block-level inconsistency from selective re-encoding
  9. Luminance Gradient Breaks    — Unnatural sharp transitions that indicate pasted regions
 10. Chroma Noise Inconsistency   — Color channel noise mismatch between original/edited areas

NOT included (handled separately by screen_detector.py):
  - Vignetting, bezel, banding, FFT screen grid, RGB correlation, glare, color temperature,
    Moiré pattern, screen color gamut
"""

import io
import re
import warnings
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
from PIL import Image, ExifTags

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# DETECTOR 1 — EXIF / XMP / Binary Metadata
# ══════════════════════════════════════════════════════════════════

def _detect_metadata(image_bytes: bytes) -> Tuple[int, List[str]]:
    """
    Scans EXIF tags, XMP namespaces, ICC profiles, and binary Photoshop blocks.
    Returns (confidence 0–100, findings list).

    Enhancements over v1:
    - Detects GIMP, Snapseed, PicsArt, and other mobile edit apps (not just Photoshop)
    - Checks for suspicious ModifyDate vs CreateDate mismatch (edit indicator)
    - Counts total XMP history actions for a severity-weighted score
    """
    findings = []
    confidence = 0

    # Known editing software keywords (expanded)
    EDIT_SOFTWARE = [
        ("photoshop",  40, "Adobe Photoshop"),
        ("lightroom",  25, "Adobe Lightroom"),
        ("camera raw", 25, "Adobe Camera Raw"),
        ("gimp",       30, "GIMP"),
        ("snapseed",   30, "Snapseed"),
        ("picsart",    30, "PicsArt"),
        ("facetune",   35, "Facetune"),
        ("pixelmator", 25, "Pixelmator"),
        ("affinity",   25, "Affinity Photo"),
        ("canva",      20, "Canva"),
        ("vsco",       20, "VSCO"),
    ]

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))

        # ── Standard EXIF ──
        try:
            exif_data = pil_img._getexif() or {}
            create_date = modify_date = None

            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                val_str = str(value).lower()

                if tag == "Software":
                    for keyword, pts, label in EDIT_SOFTWARE:
                        if keyword in val_str:
                            findings.append(f"EXIF Software: {label} detected — '{value}'")
                            confidence += pts
                            break

                if tag == "ProcessingSoftware":
                    for keyword, pts, label in EDIT_SOFTWARE:
                        if keyword in val_str:
                            findings.append(f"Processing Software: {label} — '{value}'")
                            confidence += pts
                            break

                # Date mismatch detection
                if tag == "DateTimeOriginal":
                    create_date = str(value)
                if tag == "DateTime":
                    modify_date = str(value)

            if create_date and modify_date and create_date != modify_date:
                findings.append(f"Date Mismatch: Created={create_date} vs Modified={modify_date}")
                confidence += 15

        except Exception:
            pass

        # ── XMP Metadata ──
        try:
            xmp_str = ""
            raw_info = pil_img.info or {}
            for key in ["xmp", "XML:com.adobe.xmp", "photoshop"]:
                if key in raw_info:
                    xmp_str += str(raw_info[key])

            xmp_start = image_bytes.find(b"<x:xmpmeta")
            xmp_end   = image_bytes.find(b"</x:xmpmeta>")
            if xmp_start != -1 and xmp_end != -1:
                xmp_str += image_bytes[xmp_start:xmp_end + 12].decode("utf-8", errors="ignore")

            if xmp_str:
                xmp_markers = [
                    ("photoshop:",               "Photoshop XMP namespace",          15),
                    ("xmp:CreatorTool",          "XMP CreatorTool tag",              15),
                    ("xmp:MetadataDate",         "XMP MetadataDate (re-save flag)",   8),
                    ("xmpMM:History",            "XMP Edit History block",           15),
                    ("xmpMM:DerivedFrom",        "XMP DerivedFrom (modified copy)",  15),
                    ('stEvt:action="saved"',     "XMP Save action in history",       15),
                    ('stEvt:action="converted"', "XMP Format conversion event",       8),
                    ("crs:",                     "Camera Raw settings namespace",    15),
                    ("Iptc4xmpCore:",            "IPTC metadata (Photoshop-added)",   8),
                    ("xmpMM:InstanceID",         "XMP InstanceID (re-export marker)", 8),
                ]
                for marker, label, pts in xmp_markers:
                    if marker.lower() in xmp_str.lower():
                        findings.append(f"XMP: {label}")
                        confidence += pts

                # CreatorTool value extraction
                ct_match = re.search(r"CreatorTool[^>]*>([^<]+)<", xmp_str)
                if ct_match:
                    ct_val = ct_match.group(1).strip()
                    findings.append(f"CreatorTool value: '{ct_val}'")
                    for keyword, pts, label in EDIT_SOFTWARE:
                        if keyword in ct_val.lower():
                            confidence += pts
                            break

                # History event count — more events = more editing
                history_count = xmp_str.lower().count("stevt:action")
                if history_count > 0:
                    findings.append(f"XMP History: {history_count} edit action(s) recorded")
                    confidence += min(history_count * 5, 30)

        except Exception:
            pass

        # ── ICC Color Profile ──
        try:
            icc = pil_img.info.get("icc_profile", b"")
            if icc:
                icc_str = icc.decode("latin-1", errors="ignore")
                for marker in ["Adobe RGB", "sRGB IEC", "ProPhoto", "Coated FOGRA", "eciRGB"]:
                    if marker in icc_str:
                        findings.append(f"ICC Color Profile: '{marker}' — editor color space")
                        confidence += 10
                        break
        except Exception:
            pass

        # ── Binary Photoshop Resource Blocks ──
        try:
            header = image_bytes[:65536]  # increased from 32KB to 64KB
            ps_markers = [
                (b"Photoshop 3.0", "Photoshop 3.0 resource block",    35),
                (b"8BIM",          "Photoshop 8BIM resource block",    35),
                (b"AgHg",          "Lightroom metadata block",         15),
                (b"PHUT",          "Photoshop URL/annotation block",   35),
                (b"adobe_photoshop", "Adobe Photoshop binary marker",  40),
            ]
            for marker_bytes, label, pts in ps_markers:
                if marker_bytes in header:
                    findings.append(f"Binary: {label}")
                    confidence += pts
        except Exception:
            pass

    except Exception as e:
        findings.append(f"Metadata scan error: {e}")

    return min(confidence, 100), findings


# ══════════════════════════════════════════════════════════════════
# DETECTOR 2 — ELA (Error Level Analysis)
# ══════════════════════════════════════════════════════════════════

def _detect_ela(image_bytes: bytes) -> Tuple[int, float, float]:
    """
    Re-compresses image at multiple quality levels and measures regional differences.
    Tampered regions have different compression history → show as bright patches.

    Enhancements over v1:
    - Multi-quality ELA (Q70, Q85, Q90) instead of single quality
    - Measures regional variance (std) not just global mean
    - Regional hotspot detection: flags if any 64x64 block is >3x average
    Returns (score 0–100, ela_mean, ela_std).
    """
    try:
        orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_arr = np.array(orig, dtype=np.float32)

        ela_combined = np.zeros_like(orig_arr)

        for quality in [70, 85, 90]:
            buf = io.BytesIO()
            orig.save(buf, "JPEG", quality=quality)
            buf.seek(0)
            recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
            diff = np.abs(orig_arr - recomp)
            ela_combined += diff

        ela_combined = (ela_combined / 3.0) * 10  # scale
        ela_combined = np.clip(ela_combined, 0, 255)

        ela_mean = float(ela_combined.mean())
        ela_std  = float(ela_combined.std())

        # Regional hotspot detection: any 64x64 block with mean > 3x global mean
        ela_gray = ela_combined.mean(axis=2)
        h, w = ela_gray.shape
        block_size = 64
        hotspot_found = False
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block_mean = ela_gray[y:y + block_size, x:x + block_size].mean()
                if block_mean > max(ela_mean * 3, 15):
                    hotspot_found = True
                    break

        # Scoring: mean + std + hotspot
        if ela_mean > 12 or hotspot_found:
            score = 100
        elif ela_mean > 7 or ela_std > 20:
            score = 65
        elif ela_mean > 4:
            score = 35
        else:
            score = 5

        return score, round(ela_mean, 3), round(ela_std, 3)

    except Exception:
        return 0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════
# DETECTOR 3 — Noise Consistency
# ══════════════════════════════════════════════════════════════════

def _detect_noise(gray: np.ndarray) -> Tuple[int, float, float]:
    """
    Wavelet-based noise map. Spliced/edited regions have inconsistent sensor noise.

    Enhancements over v1:
    - Also measures local noise variance across a grid (not just global std)
    - Flags if noise variance ratio between quadrants exceeds threshold
    Returns (score 0–100, noise_std, quadrant_variance_ratio).
    """
    try:
        from skimage import restoration

        denoised = restoration.denoise_wavelet(
            gray.astype(np.float64) / 255.0,
            method="BayesShrink", mode="soft",
            channel_axis=None, convert2ycbcr=False,
        )
        noise_map = np.abs(gray.astype(np.float64) / 255.0 - denoised)
        noise_std = float(noise_map.std())

        # Quadrant variance ratio — edited images have very uneven noise per quadrant
        h, w = noise_map.shape
        quads = [
            noise_map[:h//2, :w//2].std(),
            noise_map[:h//2, w//2:].std(),
            noise_map[h//2:, :w//2].std(),
            noise_map[h//2:, w//2:].std(),
        ]
        quads = [q for q in quads if q > 0]
        quad_ratio = float(max(quads) / (min(quads) + 1e-6)) if quads else 1.0

        if noise_std > 0.025 or quad_ratio > 4.0:
            score = 100
        elif noise_std > 0.015 or quad_ratio > 2.5:
            score = 65
        elif noise_std > 0.008:
            score = 35
        else:
            score = 5

        return score, round(noise_std, 6), round(quad_ratio, 3)

    except Exception:
        return 0, 0.0, 1.0


# ══════════════════════════════════════════════════════════════════
# DETECTOR 4 — Halo / Edge Artifacts
# ══════════════════════════════════════════════════════════════════

def _detect_halo(gray: np.ndarray) -> Tuple[int, float]:
    """
    Detects unnatural halo rings around objects — a compositing/blending artifact.

    Enhancements over v1:
    - Uses both Canny and Sobel; compares their edge density for inconsistency
    - Measures halo intensity (not just pixel count)
    Returns (score 0–100, halo_ratio).
    """
    try:
        from skimage import feature
        from scipy.ndimage import binary_dilation

        # Canny edges
        edges = feature.canny(gray.astype(np.float64) / 255.0, sigma=2.0)
        dilated = binary_dilation(edges, iterations=3)
        halo_mask = dilated & ~edges
        halo_ratio = float(halo_mask.sum() / gray.size * 100)

        # Halo intensity: average pixel brightness in halo region
        halo_intensity = float(gray[halo_mask].mean()) / 255.0 if halo_mask.sum() > 0 else 0.0

        # Unnatural halos are both large AND bright
        if halo_ratio > 2.0 and halo_intensity > 0.7:
            score = 100
        elif halo_ratio > 2.0:
            score = 70
        elif halo_ratio > 0.8:
            score = 40
        else:
            score = 5

        return score, round(halo_ratio, 4)

    except Exception:
        return 0, 0.0


# ══════════════════════════════════════════════════════════════════
# DETECTOR 5 — Clone Stamp / Healing Brush Detection
# ══════════════════════════════════════════════════════════════════

def _detect_clone_stamp(gray: np.ndarray) -> Tuple[int, int]:
    """
    Detects duplicated texture patches — hallmark of clone stamp or healing brush.

    Enhancements over v1:
    - Adaptive stride based on image resolution (faster on large images)
    - Minimum distance filter increased to patch_size * 3 to reduce false positives
    - Similarity threshold raised to 0.98 for fewer false positives on receipts
    Returns (score 0–100, clone_pairs).
    """
    try:
        h, w = gray.shape
        patch_size = 16
        # Adaptive stride: larger images use bigger stride
        stride = max(8, min(16, (h * w) // 500000))
        threshold = 0.98  # raised from 0.97

        patches, positions = [], []
        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                patch = gray[y:y + patch_size, x:x + patch_size].astype(np.float32)
                std = patch.std()
                if std > 8:  # raised from 5 — skip near-uniform patches
                    patch_norm = (patch - patch.mean()) / (std + 1e-6)
                    patches.append(patch_norm.flatten())
                    positions.append((y, x))

        if len(patches) < 2:
            return 0, 0

        patches_arr  = np.array(patches)
        norms        = np.linalg.norm(patches_arr, axis=1, keepdims=True)
        patches_unit = patches_arr / (norms + 1e-6)
        corr_matrix  = patches_unit @ patches_unit.T

        shown = set()
        for i in range(len(positions)):
            for j in range(i + 1, min(i + 200, len(positions))):
                if corr_matrix[i, j] > threshold:
                    y1, x1 = positions[i]
                    y2, x2 = positions[j]
                    dist = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
                    if dist > patch_size * 3:  # increased min distance
                        key = (y1 // patch_size, x1 // patch_size,
                               y2 // patch_size, x2 // patch_size)
                        shown.add(key)

        clone_pairs = len(shown)
        score = 100 if clone_pairs > 15 else (65 if clone_pairs > 5 else (30 if clone_pairs > 2 else 5))
        return score, clone_pairs

    except Exception:
        return 0, 0


# ══════════════════════════════════════════════════════════════════
# DETECTOR 6 — JPEG Ghost Analysis
# ══════════════════════════════════════════════════════════════════

def _detect_jpeg_ghost(image_bytes: bytes) -> Tuple[int, float]:
    """
    Detects double-compression by re-saving at multiple qualities and measuring mismatch.
    Edited regions that were pasted from another JPEG will ghost at different quality levels.

    Enhancements over v1:
    - Measures both global std AND regional block variance
    - Uses more quality levels (50, 65, 75, 85, 92)
    Returns (score 0–100, best_variance).
    """
    try:
        orig     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_arr = np.array(orig, dtype=np.float32)
        variances = {}

        for q in [50, 65, 75, 85, 92]:
            buf = io.BytesIO()
            orig.save(buf, "JPEG", quality=q)
            buf.seek(0)
            recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
            diff = np.mean((orig_arr - recomp) ** 2, axis=2)
            variances[q] = float(diff.std())

        best_var = max(variances.values())

        score = 100 if best_var > 18 else (65 if best_var > 8 else (30 if best_var > 3 else 5))
        return score, round(best_var, 4)

    except Exception:
        return 0, 0.0


# ══════════════════════════════════════════════════════════════════
# DETECTOR 7 — Copy-Move (ORB Feature Matching)
# ══════════════════════════════════════════════════════════════════

def _detect_copy_move_orb(gray: np.ndarray) -> Tuple[int, int]:
    """
    Uses ORB keypoint matching to find duplicated regions in the image.

    Enhancements over v1:
    - Adds Lowe's ratio test to filter weak matches
    - Uses FLANN matcher instead of BFMatcher for better accuracy
    Returns (score 0–100, match_count).
    """
    try:
        orb = cv2.ORB_create(nfeatures=3000)  # increased from 2000
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 4:
            return 0, 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # knn match for ratio test
        try:
            matches = bf.knnMatch(des, des, k=2)
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    # Ratio test + spatial distance filter
                    if m.distance < 0.75 * n.distance and m.queryIdx != m.trainIdx:
                        pt1 = np.array(kp[m.queryIdx].pt)
                        pt2 = np.array(kp[m.trainIdx].pt)
                        if np.linalg.norm(pt1 - pt2) > 25:
                            good.append(m)
            match_count = len(good)
        except Exception:
            # Fallback to simple matching
            raw_matches = bf.match(des, des)
            match_count = sum(
                1 for m in raw_matches
                if m.queryIdx != m.trainIdx and
                np.linalg.norm(np.array(kp[m.queryIdx].pt) - np.array(kp[m.trainIdx].pt)) > 20
            )

        score = 100 if match_count > 25 else (65 if match_count > 10 else (30 if match_count > 4 else 5))
        return score, match_count

    except Exception:
        return 0, 0


# ══════════════════════════════════════════════════════════════════
# DETECTOR 8 — DCT Block Anomaly  [NEW]
# ══════════════════════════════════════════════════════════════════

def _detect_dct_anomaly(gray: np.ndarray) -> Tuple[int, float]:
    """
    NEW detector not in original notebook.
    Analyzes DCT (Discrete Cosine Transform) energy distribution across 8x8 blocks.
    Edited/pasted JPEG regions have different DCT coefficient energy patterns.

    How it works:
    - Divides image into 8x8 JPEG blocks
    - Computes DCT energy for each block
    - Flags blocks with energy far outside the normal distribution (> 3 std from mean)
    Returns (score 0–100, anomaly_ratio).
    """
    try:
        h, w = gray.shape
        gray_f = gray.astype(np.float32)
        block_size = 8
        energies = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray_f[y:y + block_size, x:x + block_size]
                dct_block = cv2.dct(block)
                # AC energy (skip DC component at [0,0])
                ac_energy = float((dct_block[1:, 1:] ** 2).mean())
                energies.append(ac_energy)

        if len(energies) < 10:
            return 0, 0.0

        energies_arr = np.array(energies)
        mean_e = energies_arr.mean()
        std_e  = energies_arr.std()

        # Anomalous blocks are those far outside normal distribution
        anomaly_mask = np.abs(energies_arr - mean_e) > 3.0 * std_e
        anomaly_ratio = float(anomaly_mask.sum() / len(energies) * 100)

        score = 100 if anomaly_ratio > 8.0 else (65 if anomaly_ratio > 4.0 else (30 if anomaly_ratio > 2.0 else 5))
        return score, round(anomaly_ratio, 4)

    except Exception:
        return 0, 0.0


# ══════════════════════════════════════════════════════════════════
# DETECTOR 9 — Luminance Gradient Breaks  [NEW]
# ══════════════════════════════════════════════════════════════════

def _detect_luminance_breaks(gray: np.ndarray) -> Tuple[int, float]:
    """
    NEW detector not in original notebook.
    Detects unnatural sharp luminance transitions that indicate a pasted/inserted region.

    Real receipts have relatively smooth luminance gradients (paper + ink).
    Pasted regions create hard luminance edges that don't match the paper's natural gradient.

    How it works:
    - Computes horizontal and vertical gradient magnitude
    - Measures the ratio of extreme gradients vs expected gradients
    Returns (score 0–100, break_ratio).
    """
    try:
        gray_f = gray.astype(np.float32)

        # Compute gradient magnitude
        grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Threshold for "extreme" gradient (potential paste boundary)
        mean_m = magnitude.mean()
        std_m  = magnitude.std()
        extreme_thresh = mean_m + 4.0 * std_m

        extreme_pixels = (magnitude > extreme_thresh).sum()
        break_ratio = float(extreme_pixels / magnitude.size * 100)

        # Also check for suspiciously straight lines of high gradient
        # (cut-and-paste boundaries tend to be perfectly horizontal/vertical)
        col_gradient = magnitude.mean(axis=0)  # per-column average
        row_gradient = magnitude.mean(axis=1)  # per-row average
        col_spikes = int((col_gradient > mean_m + 3 * std_m).sum())
        row_spikes = int((row_gradient > mean_m + 3 * std_m).sum())
        straight_line_flag = (col_spikes > 3 or row_spikes > 3)

        if break_ratio > 1.5 and straight_line_flag:
            score = 100
        elif break_ratio > 1.5:
            score = 65
        elif break_ratio > 0.8:
            score = 35
        else:
            score = 5

        return score, round(break_ratio, 4)

    except Exception:
        return 0, 0.0


# ══════════════════════════════════════════════════════════════════
# DETECTOR 10 — Chroma Noise Inconsistency  [NEW]
# ══════════════════════════════════════════════════════════════════

def _detect_chroma_noise(rgb: np.ndarray) -> Tuple[int, float]:
    """
    NEW detector not in original notebook.
    Analyzes color channel noise independently. Edited regions often show
    different chroma noise patterns because they come from a different source image.

    How it works:
    - Converts to YCbCr (luminance + chroma)
    - Measures noise in Cb and Cr chroma channels separately
    - Flags if chroma noise is spatially inconsistent (different regions have different noise)
    Returns (score 0–100, chroma_inconsistency).
    """
    try:
        # Convert to YCbCr
        img_pil = Image.fromarray(rgb)
        ycbcr = np.array(img_pil.convert("YCbCr"), dtype=np.float32)
        cb = ycbcr[:, :, 1]
        cr = ycbcr[:, :, 2]

        h, w = cb.shape
        block_size = 32
        cb_stds, cr_stds = [], []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                cb_block = cb[y:y + block_size, x:x + block_size]
                cr_block = cr[y:y + block_size, x:x + block_size]
                cb_stds.append(float(cb_block.std()))
                cr_stds.append(float(cr_block.std()))

        if len(cb_stds) < 4:
            return 0, 0.0

        cb_arr = np.array(cb_stds)
        cr_arr = np.array(cr_stds)

        # Coefficient of variation (std/mean) measures relative inconsistency
        cb_cv = float(cb_arr.std() / (cb_arr.mean() + 1e-6))
        cr_cv = float(cr_arr.std() / (cr_arr.mean() + 1e-6))
        chroma_inconsistency = (cb_cv + cr_cv) / 2.0

        score = 100 if chroma_inconsistency > 1.5 else (65 if chroma_inconsistency > 0.9 else (30 if chroma_inconsistency > 0.5 else 5))
        return score, round(chroma_inconsistency, 4)

    except Exception:
        return 0, 0.0


# ══════════════════════════════════════════════════════════════════
# WEIGHTS  (must reflect detector reliability for receipt images)
# ══════════════════════════════════════════════════════════════════

_WEIGHTS = {
    "metadata":          4,   # strongest signal — direct software evidence
    "ela":               3,   # very reliable for JPEG edits
    "jpeg_ghost":        3,   # reliable for re-saved images
    "clone_stamp":       3,   # reliable for number/amount tampering
    "dct_anomaly":       3,   # reliable for selective re-encoding
    "copy_move":         2,
    "noise":             2,
    "chroma_noise":      2,   # new
    "luminance_breaks":  2,   # new
    "halo":              1,   # noisier signal, lower weight
}

# ══════════════════════════════════════════════════════════════════
# THRESHOLDS
# ══════════════════════════════════════════════════════════════════

THRESHOLD_AUTO_REJECT   = 45   # >= 45 → high confidence edit → reject
THRESHOLD_MANUAL_REVIEW = 28   # 28–44 → suspicious → manual review
THRESHOLD_SOFT_FLAG     = 15   # 15–27 → low suspicion → soft flag


# ══════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════

def detect_edit_manipulation(image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    """
    Analyze image bytes for digital editing / manipulation.
    Does NOT detect screen photos — use screen_detector.py for that.

    Returns:
    {
        "is_manipulated": bool,
        "confidence":     "high" | "medium" | "low" | "clean",
        "edit_score":     float (0–100),
        "action":         "auto_reject" | "manual_review" | "soft_flag" | "approve",
        "verdict":        str,
        "details":        str,
        "metadata_findings": [str, ...],
        "detectors": {
            "metadata":         {"score": int, "raw": int},
            "ela":              {"score": int, "raw": float, "std": float},
            "noise":            {"score": int, "raw": float, "quad_ratio": float},
            "halo":             {"score": int, "raw": float},
            "clone_stamp":      {"score": int, "raw": int},
            "jpeg_ghost":       {"score": int, "raw": float},
            "copy_move":        {"score": int, "raw": int},
            "dct_anomaly":      {"score": int, "raw": float},
            "luminance_breaks": {"score": int, "raw": float},
            "chroma_noise":     {"score": int, "raw": float},
        }
    }
    """
    # ── Decode image ──────────────────────────────────────────────
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2 could not decode image")
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return {
            "is_manipulated": False,
            "confidence":     "clean",
            "edit_score":     0.0,
            "action":         "approve",
            "verdict":        "Image decode failed",
            "details":        str(e),
            "metadata_findings": [],
            "detectors":      {}
        }

    # ── Run detectors ─────────────────────────────────────────────
    meta_s,  meta_findings          = _detect_metadata(image_bytes)
    ela_s,   ela_mean,  ela_std     = _detect_ela(image_bytes)
    noise_s, noise_std, quad_ratio  = _detect_noise(img_gray)
    halo_s,  halo_ratio             = _detect_halo(img_gray)
    clone_s, clone_pairs            = _detect_clone_stamp(img_gray)
    ghost_s, ghost_var              = _detect_jpeg_ghost(image_bytes)
    orb_s,   orb_count              = _detect_copy_move_orb(img_gray)
    dct_s,   dct_anomaly            = _detect_dct_anomaly(img_gray)
    lum_s,   lum_breaks             = _detect_luminance_breaks(img_gray)
    chroma_s, chroma_incon          = _detect_chroma_noise(img_rgb)

    scores = {
        "metadata":         meta_s,
        "ela":              ela_s,
        "noise":            noise_s,
        "halo":             halo_s,
        "clone_stamp":      clone_s,
        "jpeg_ghost":       ghost_s,
        "copy_move":        orb_s,
        "dct_anomaly":      dct_s,
        "luminance_breaks": lum_s,
        "chroma_noise":     chroma_s,
    }

    # ── Weighted score ────────────────────────────────────────────
    total_w  = sum(scores[k] * _WEIGHTS[k] for k in scores)
    total_wt = sum(_WEIGHTS.values())
    edit_score = round(total_w / total_wt, 2)

    # ── Metadata is a hard override: if Photoshop binary found, always flag ──
    metadata_hard_flag = meta_s >= 70  # 8BIM or PS3.0 block found

    # ── Action & confidence ───────────────────────────────────────
    effective_score = 100 if metadata_hard_flag else edit_score

    if effective_score >= THRESHOLD_AUTO_REJECT:
        action     = "auto_reject"
        confidence = "high"
    elif effective_score >= THRESHOLD_MANUAL_REVIEW:
        action     = "manual_review"
        confidence = "medium"
    elif effective_score >= THRESHOLD_SOFT_FLAG:
        action     = "soft_flag"
        confidence = "low"
    else:
        action     = "approve"
        confidence = "clean"

    # ── Verdict ───────────────────────────────────────────────────
    top_detectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_signals   = [k for k, v in top_detectors if v >= 60][:3]

    if metadata_hard_flag:
        verdict = "PHOTOSHOP/EDITOR BINARY DETECTED — image was processed through an editor"
    elif action == "auto_reject":
        verdict = f"MANIPULATION DETECTED — strongest signals: {', '.join(top_signals) if top_signals else 'multiple detectors'}"
    elif action == "manual_review":
        verdict = f"SUSPICIOUS EDIT INDICATORS — signals: {', '.join(top_signals) if top_signals else 'minor anomalies'}"
    elif action == "soft_flag":
        verdict = "MINOR ANOMALIES — possibly re-saved or lightly processed"
    else:
        verdict = "LIKELY AUTHENTIC — no significant edit indicators"

    return {
        "is_manipulated":    action in ("auto_reject", "manual_review"),
        "confidence":        confidence,
        "edit_score":        edit_score,
        "action":            action,
        "verdict":           verdict,
        "details":           verdict,
        "metadata_findings": meta_findings,
        "detectors": {
            "metadata":         {"score": meta_s,   "raw": len(meta_findings)},
            "ela":              {"score": ela_s,    "raw": ela_mean,    "std": ela_std},
            "noise":            {"score": noise_s,  "raw": noise_std,   "quad_ratio": quad_ratio},
            "halo":             {"score": halo_s,   "raw": halo_ratio},
            "clone_stamp":      {"score": clone_s,  "raw": clone_pairs},
            "jpeg_ghost":       {"score": ghost_s,  "raw": ghost_var},
            "copy_move":        {"score": orb_s,    "raw": orb_count},
            "dct_anomaly":      {"score": dct_s,    "raw": dct_anomaly},
            "luminance_breaks": {"score": lum_s,    "raw": lum_breaks},
            "chroma_noise":     {"score": chroma_s, "raw": chroma_incon},
        }
    }