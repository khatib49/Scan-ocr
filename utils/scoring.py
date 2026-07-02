# utils/scoring.py
"""
Deterministic fraud-scoring engine (flow v2).

Replaces model-guessed fraudScore with a rule-based, reproducible score.
The LLM's own fraud estimate is only a minor signal (20% weight).

Signals:
  QR_TOTAL_MISMATCH        ZATCA QR total != extracted total          (+45)
  QR_VAT_NUMBER_MISMATCH   ZATCA QR VAT number != extracted TaxID     (+30)
  QR_SELLER_MISMATCH       ZATCA QR seller name unlike merchant name  (+20)
  MATH_MISMATCH            Subtotal + Tax != Total (±1 SAR)           (+30)
  VAT_RATE_OFF             Tax not ~15% of Subtotal (±2%)             (+20)
  INVOICE_ID_MISSING       No invoice ID extracted                    (+40)
  TAXID_PROFILE_MISMATCH   TaxID != profile's expected TaxID          (+30)
  SCREEN_FLAGGED           Screen detector in manual-review band      (+15)

QR agreement (total matches) is a strong authenticity signal:
it caps the final fraud score at 40 unless the screen detector flagged.
"""

from typing import Any, Dict, Optional

from app.text_utils import fuzzy_ratio

WEIGHTS = {
    "QR_TOTAL_MISMATCH": 45,
    "QR_VAT_NUMBER_MISMATCH": 30,
    "QR_SELLER_MISMATCH": 20,
    "MATH_MISMATCH": 30,
    "VAT_RATE_OFF": 20,
    "INVOICE_ID_MISSING": 40,
    "TAXID_PROFILE_MISMATCH": 30,
    "SCREEN_FLAGGED": 15,
}

MODEL_ESTIMATE_WEIGHT = 0.20   # LLM's own fraud guess contributes at most 20 pts
MATH_TOLERANCE_SAR = 1.0
QR_TOTAL_TOLERANCE_SAR = 1.0
VAT_TARGET = 0.15
VAT_TOLERANCE = 0.02
QR_SELLER_MIN_SIM = 0.45       # lenient: QR seller is legal name, receipt shows brand
QR_MATCH_SCORE_CAP = 40        # verified QR caps fraud unless screen-flagged


def _num(x) -> Optional[float]:
    try:
        return float(str(x).replace(",", "").replace("SAR", "").strip())
    except Exception:
        return None


def _digits(x) -> str:
    return "".join(ch for ch in str(x or "") if ch.isdigit())


def deterministic_score(
    fields: Dict[str, Any],
    profile: Optional[Dict[str, Any]],
    qr: Optional[Dict[str, Any]],
    screen_result: Optional[Dict[str, Any]],
    model_fraud_estimate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    fields: extracted invoice fields (MerchantName, Total, Tax, ...)
    profile: matched venue profile (or None)
    qr: decoded ZATCA QR dict {seller, vat, timestamp, total, vat_amount} (or None)
    screen_result: screen detector output (or None)

    Returns: {fraudScore, confidentScore, reasons, signals, qrVerified, needsRescan}
    """
    reasons: list[str] = []
    signals: list[str] = []
    score = 0.0

    subtotal = _num(fields.get("Subtotal"))
    tax = _num(fields.get("Tax"))
    total = _num(fields.get("Total"))
    discount = _num(fields.get("Discount"))

    qr_verified: Optional[bool] = None  # None = no QR on receipt

    # ── 1) ZATCA QR cross-checks (ground truth when present) ──────
    if qr:
        qr_total = qr.get("total")
        qr_vat_no = qr.get("vat")
        qr_seller = qr.get("seller")

        if qr_total is not None and total is not None:
            if abs(qr_total - total) > QR_TOTAL_TOLERANCE_SAR:
                score += WEIGHTS["QR_TOTAL_MISMATCH"]
                signals.append("QR_TOTAL_MISMATCH")
                reasons.append(
                    f"ZATCA QR total ({qr_total}) does not match extracted total ({total})"
                )
                qr_verified = False
            else:
                qr_verified = True

        if qr_vat_no and fields.get("TaxID"):
            if _digits(qr_vat_no) != _digits(fields["TaxID"]):
                score += WEIGHTS["QR_VAT_NUMBER_MISMATCH"]
                signals.append("QR_VAT_NUMBER_MISMATCH")
                reasons.append("ZATCA QR VAT number does not match extracted TaxID")
                qr_verified = False

        if qr_seller and fields.get("MerchantName"):
            sim = fuzzy_ratio(str(qr_seller), str(fields["MerchantName"]))
            if sim < QR_SELLER_MIN_SIM:
                score += WEIGHTS["QR_SELLER_MISMATCH"]
                signals.append("QR_SELLER_MISMATCH")
                reasons.append(
                    f"ZATCA QR seller name unlike extracted merchant name (sim={sim:.2f})"
                )

    # ── 2) Math validation ─────────────────────────────────────────
    if subtotal is not None and tax is not None and total is not None:
        # tax-inclusive: subtotal ≈ total; tax-exclusive: subtotal + tax ≈ total
        exclusive_ok = abs((subtotal + tax) - total) <= MATH_TOLERANCE_SAR
        inclusive_ok = abs(subtotal - total) <= MATH_TOLERANCE_SAR
        with_discount_ok = (
            discount is not None
            and abs((subtotal - discount + tax) - total) <= MATH_TOLERANCE_SAR
        )
        if not (exclusive_ok or inclusive_ok or with_discount_ok):
            score += WEIGHTS["MATH_MISMATCH"]
            signals.append("MATH_MISMATCH")
            reasons.append(
                f"Math check failed: Subtotal({subtotal}) + Tax({tax}) != Total({total})"
            )

        # KSA VAT ≈ 15%
        base = subtotal - (discount or 0)
        if base and base > 0.1:
            observed = tax / base
            inclusive_observed = tax / (base - tax) if (base - tax) > 0.1 else None
            ok = abs(observed - VAT_TARGET) <= VAT_TOLERANCE or (
                inclusive_observed is not None
                and abs(inclusive_observed - VAT_TARGET) <= VAT_TOLERANCE
            )
            if not ok:
                score += WEIGHTS["VAT_RATE_OFF"]
                signals.append("VAT_RATE_OFF")
                reasons.append(f"VAT rate {observed:.1%} deviates from 15%")

    # ── 3) Required fields ─────────────────────────────────────────
    if not fields.get("InvoiceId"):
        score += WEIGHTS["INVOICE_ID_MISSING"]
        signals.append("INVOICE_ID_MISSING")
        reasons.append("InvoiceId missing or unreadable")

    # ── 4) Profile checks ──────────────────────────────────────────
    if profile and fields.get("TaxID") and profile.get("TaxID_Label"):
        if _digits(fields["TaxID"]) != _digits(profile["TaxID_Label"]):
            score += WEIGHTS["TAXID_PROFILE_MISMATCH"]
            signals.append("TAXID_PROFILE_MISMATCH")
            reasons.append("TaxID does not match the merchant profile")

    # ── 5) Screen-photo flag (manual-review band) ──────────────────
    screen_flagged = bool(screen_result and screen_result.get("action") == "manual_review")
    if screen_flagged:
        score += WEIGHTS["SCREEN_FLAGGED"]
        signals.append("SCREEN_FLAGGED")
        reasons.append("Possible screen-photographed receipt")

    # ── 6) Model estimate as minor signal ──────────────────────────
    if model_fraud_estimate is not None:
        try:
            score += max(0.0, min(100.0, float(model_fraud_estimate))) * MODEL_ESTIMATE_WEIGHT
        except Exception:
            pass

    # QR agreement is strong authenticity evidence — cap unless screen-flagged
    if qr_verified is True and not screen_flagged:
        score = min(score, QR_MATCH_SCORE_CAP)

    fraud = int(max(0, min(100, round(score))))

    # Confidence: field completeness + QR verification
    key_fields = ["MerchantName", "TransactionDate", "InvoiceId", "TaxID", "Subtotal", "Tax", "Total"]
    present = sum(1 for k in key_fields if fields.get(k) not in (None, "", "null"))
    confident = int(round(100 * present / len(key_fields)))
    if qr_verified is True:
        confident = min(100, confident + 15)
    elif qr_verified is False:
        confident = max(0, confident - 30)

    return {
        "fraudScore": fraud,
        "confidentScore": confident,
        "reasons": reasons,
        "signals": signals,
        "qrVerified": qr_verified,
        "needsRescan": total is None or fraud >= 80,
    }
