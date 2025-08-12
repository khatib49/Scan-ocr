from typing import Optional, Dict, Any
from datetime import datetime

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

def validate_and_score(data: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # You can keep your original logic here
    d = data.get("data", {})
    subtotal = coerce_number(d.get("Subtotal"))
    tax = coerce_number(d.get("Tax"))
    total = coerce_number(d.get("Total"))
    reason = []

    if subtotal is not None and tax is not None and total is not None:
        expected = round(subtotal + tax, 2)
        if abs(expected - total) > 0.1:
            reason.append("Subtotal + Tax != Total")

    if profile:
        if d.get("TaxID") and profile.get("TaxID_Label") and d.get("TaxID") != profile["TaxID_Label"]:
            reason.append("TaxID mismatch")
        if d.get("MerchantName") and profile.get("MerchantName_Keyword"):
            if not any(k.lower() in d["MerchantName"].lower() for k in profile["MerchantName_Keyword"]):
                reason.append("Merchant name mismatch")

    d["fraudScore"] = 100 if reason else 0
    d["confidentScore"] = 0 if reason else 100
    d["reason"] = ", ".join(reason) if reason else "All values match the venue profile and calculations are correct."
    data["data"] = d
    return data
