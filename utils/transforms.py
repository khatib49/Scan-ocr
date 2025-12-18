from typing import Optional, Dict, Any
from datetime import datetime

from app.text_utils import fuzzy_ratio, jaccard, tokenize_distinct

def coerce_number(x):
    try:
        return float(str(x).replace(",", "").replace("SAR", ""))
    except:
        return None

def coerce_nullish(x):
    if x is None:
        return None
    x = str(x).strip()
    return None if not x or x.lower() in ("null", "none", "n/a", "-") else x

def norm_date(datestr: Optional[str]) -> Optional[str]:
    if not datestr:
        return None
    try:
        return datetime.fromisoformat(datestr).isoformat()
    except Exception:
        return None

def validate_and_score(
    data: Dict[str, Any],
    profile: Optional[Dict[str, Any]],
    image_url: Optional[str],
    merchant_guess: Optional[str] = None,
    matched: Optional[bool] = None
) -> Dict[str, Any]:
    d = data.get("data", {}) or {}
    subtotal = coerce_number(d.get("Subtotal"))
    tax = coerce_number(d.get("Tax"))
    total = coerce_number(d.get("Total"))
    discount = coerce_number(d.get("Discount"))
    reason = []

    MATH_TOLERANCE_SAR = 1.50     # Increased tolerance for rounding
    VAT_TARGET         = 0.15
    VAT_TOLERANCE      = 0.020    # Increased to 2% for tax-inclusive scenarios
    SMALL_TOL          = 0.10
    fraud = 0
    confident = 100
    math_ok = None
    vat_ok  = None

    # Check if profile indicates tax-inclusive
    tax_inclusive = False
    if profile:
        hints = profile.get("ExtractionHints") or {}
        tax_inclusive = hints.get("TaxInclusive", False)

    if subtotal is not None and tax is not None and total is not None:
        
        # IMPORTANT: Subtotal from receipt is AFTER discount and may be tax-inclusive
        if tax_inclusive:
            # Tax-inclusive scenario: Subtotal already includes tax
            # Formula: net = subtotal / 1.15, tax_check = net * 0.15
            net_amount = subtotal / 1.15
            expected_tax = net_amount * 0.15
            
            # Check if tax matches
            vat_ok = abs(expected_tax - tax) <= (expected_tax * VAT_TOLERANCE)
            
            # Check if total matches subtotal (since tax is included)
            math_ok = abs(subtotal - total) <= MATH_TOLERANCE_SAR
            
        elif discount is not None and discount > 0:
            # Discount scenario (tax-exclusive)
            # The subtotal is AFTER discount, so we validate:
            # subtotal + tax = total
            expected_total = round(subtotal + tax, 2)
            math_ok = abs(expected_total - total) <= MATH_TOLERANCE_SAR

            # VAT check on post-discount subtotal
            if subtotal > SMALL_TOL:
                observed_vat_rate = tax / subtotal
                vat_ok = abs(observed_vat_rate - VAT_TARGET) <= VAT_TOLERANCE

        else:
            # Normal scenario (no discount, tax-exclusive)
            expected_total = round(subtotal + tax, 2)
            math_ok = abs(expected_total - total) <= MATH_TOLERANCE_SAR

            # VAT normal
            if subtotal > SMALL_TOL:
                observed_vat_rate = tax / subtotal
                vat_ok = abs(observed_vat_rate - VAT_TARGET) <= VAT_TOLERANCE

    # Profile-based checks
    name_mismatch = False
    if profile:
        # TaxID exact label match
        if d.get("TaxID") and profile.get("TaxID_Label") and d.get("TaxID") != profile["TaxID_Label"]:
            fraud += 30
            confident -= 30
            reason.append("TaxID mismatch")

        # Merchant name agreement
        prof_keywords = profile.get("MerchantName_Keyword") or []
        if d.get("MerchantName") and isinstance(prof_keywords, list) and prof_keywords:
            observed_name = str(d["MerchantName"]).strip()
            obs_tokens = set(tokenize_distinct(observed_name))

            best_fuzzy = 0.0
            best_kw = None
            for kw in prof_keywords:
                fz = fuzzy_ratio(observed_name, kw or "")
                if fz > best_fuzzy:
                    best_fuzzy, best_kw = fz, kw

            has_overlap = False
            for kw in prof_keywords:
                kw_tokens = set(tokenize_distinct(kw or ""))
                if kw_tokens and (obs_tokens & kw_tokens):
                    has_overlap = True
                    break

            MIN_FUZZY = 0.80
            if (best_fuzzy < MIN_FUZZY) or (not has_overlap):
                name_mismatch = True
                fraud = 100
                confident = 0
                reason.append("Merchant name mismatch")

    if math_ok is False:
        reason.append("Math check failed")
        fraud = 100

    if vat_ok is False:
        reason.append("VAT check failed")
        fraud = 100
        
    # If both pass, set good scores
    if math_ok and vat_ok and not name_mismatch:
        fraud = 0
        confident = 95

    d["fraudScore"] = fraud
    d["confidentScore"] = confident
    if reason:
        d["reason"] = ", ".join(reason)
    else:
        d["reason"] = "All validations passed" if fraud == 0 else "No issues detected"

    d["image_url"] = image_url
    d["profileMatched"] = bool(matched) if merchant_guess else False if name_mismatch else bool(matched)
    d["needsRescan"] = (
        (merchant_guess is None or str(merchant_guess).strip() == "") or
        (not matched) or
        name_mismatch or
        (total is None)
    )

    data["data"] = d
    return data