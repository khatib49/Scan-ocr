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
    reason = []

    # 1) Math sanity: Subtotal + Tax ≈ Total
    if subtotal is not None and tax is not None and total is not None:
        expected = round(subtotal + tax, 2)
        if abs(expected - total) > 0.10:
            reason.append("Subtotal + Tax != Total")

    # 2) Profile-based checks
    name_mismatch = False
    if profile:
        # TaxID exact label match (if both present)
        if d.get("TaxID") and profile.get("TaxID_Label") and d.get("TaxID") != profile["TaxID_Label"]:
            reason.append("TaxID mismatch")

        # Merchant name agreement (ignore generic words; allow AR/EN normalization)
        prof_keywords = profile.get("MerchantName_Keyword") or []
        if d.get("MerchantName") and isinstance(prof_keywords, list) and prof_keywords:
            # Token / fuzzy gates
            observed_name = str(d["MerchantName"]).strip()
            obs_tokens = set(tokenize_distinct(observed_name))

            # choose the best keyword by fuzzy similarity
            best_fuzzy = 0.0
            best_kw = None
            for kw in prof_keywords:
                fz = fuzzy_ratio(observed_name, kw or "")
                if fz > best_fuzzy:
                    best_fuzzy, best_kw = fz, kw

            # also require at least one non-generic token overlap
            has_overlap = False
            for kw in prof_keywords:
                kw_tokens = set(tokenize_distinct(kw or ""))
                if kw_tokens and (obs_tokens & kw_tokens):
                    has_overlap = True
                    break

            # tuneable thresholds
            MIN_FUZZY = 0.80  # conservative; adjust with data
            if (best_fuzzy < MIN_FUZZY) or (not has_overlap):
                name_mismatch = True
                reason.append("Merchant name mismatch")

    # 3) Fraud/Confidence + bookkeeping
    d["fraudScore"] = 100 if reason else 0
    d["confidentScore"] = 0 if reason else 100
    if reason:
        d["reason"] = ", ".join(reason)

    d["image_url"] = image_url

    # 4) Profile match flag:
    #    If name mismatch is detected, we do NOT trust the profile even if upstream matched=True.
    if name_mismatch:
        d["profileMatched"] = False
    else:
        d["profileMatched"] = bool(matched) if merchant_guess else False

    # 5) needsRescan rules:
    #    True when: no guess, not matched, name mismatch, or Total missing (critical value).
    d["needsRescan"] = (
        (merchant_guess is None or str(merchant_guess).strip() == "") or
        (not matched) or
        name_mismatch or
        (total is None)
    )

    data["data"] = d
    return data