import math, re, unicodedata
from datetime import datetime
from typing import Optional, Dict, Any

# --- Number parsing and coercion ---

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
NULL_STRINGS = {"", "null", "none", "nil", "n/a", "na", "—", "-", "غير متوفر", "غير موجود"}


def coerce_number(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x).replace(",", "").replace("SAR", "").strip()
        s = s.translate(ARABIC_DIGITS)
        return float(s)
    except:
        return None


def coerce_nullish(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    return None if s in NULL_STRINGS else x


# --- Date normalization ---

def norm_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip().replace("  ", " ").translate(ARABIC_DIGITS)

    patterns = [
        "%d/%m/%Y %I:%M %p", "%d/%m/%Y %H:%M",
        "%Y/%m/%d %I:%M:%S %p", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
        "%d/%m/%Y", "%Y/%m/%d",
        "%d-%m-%Y %H:%M", "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    ]
    for fmt in patterns:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            continue

    try:
        iso_like = s.replace("T", " ").replace("Z", "").split(".")[0]
        dt = datetime.fromisoformat(iso_like)
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return s  # fallback


# --- Validation & Scoring ---

def validate_and_score(payload: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    d = payload.get("data", {})
    issues = []
    fraud = 0
    conf = 30

    subtotal = coerce_number(d.get("Subtotal"))
    tax = coerce_number(d.get("Tax"))
    total = coerce_number(d.get("Total"))

    if subtotal is not None and tax is not None and total is not None:
        if abs((subtotal + tax) - total) > 1:
            fraud += 20
            issues.append("Subtotal + Tax != Total")
        else:
            conf += 15

    if subtotal and tax:
        vat_expected = subtotal * 0.15
        if abs(tax - vat_expected) > subtotal * 0.015:
            fraud += 15
            issues.append("Tax not ≈ 15% of Subtotal")
        else:
            conf += 15

    if profile:
        m_match = _fuzzy_match(profile.get("MerchantName_Keyword"), d.get("MerchantName"))
        a_match = _fuzzy_match(profile.get("MerchantAddress_Keyword"), d.get("MerchantAddress"))

        if m_match < 70:
            fraud += 15
            issues.append("Merchant name mismatch")
        else:
            conf += 10

        if a_match < 60:
            fraud += 10
            issues.append("Address mismatch")
        else:
            conf += 10

    d["fraudScore"] = min(100, fraud)
    d["confidentScore"] = min(100, conf)
    d["reason"] = "; ".join(issues) if issues else "All values validated."
    return {"data": d}


def _fuzzy_match(candidates, value):
    import unicodedata
    from rapidfuzz import fuzz

    def normalize(text):
        text = text or ""
        text = unicodedata.normalize("NFKC", text.lower())
        text = re.sub(r"[\s\W]+", " ", text)
        return text.strip()

    if not candidates or not value:
        return 0
    value_norm = normalize(value)
    if isinstance(candidates, list):
        return max(fuzz.partial_ratio(normalize(c), value_norm) for c in candidates if c)
    return fuzz.partial_ratio(normalize(str(candidates)), value_norm)
