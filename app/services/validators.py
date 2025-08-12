from typing import Dict, Any, List

def safe_float(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

VAT_KSA = 0.15

# You can import your own matcher if available
try:
    from ..services.venue_matcher import match_venue_profile
except Exception:
    def match_venue_profile(_data):
        return {"matched": False, "profile": None, "hints": {}}


def validate_extracted(data: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    checks: Dict[str, Any] = {}

    subtotal = safe_float(data.get("Subtotal"))
    tax = safe_float(data.get("Tax"))
    total = safe_float(data.get("Total"))

    # Sum check
    if subtotal is not None and tax is not None and total is not None:
        sum_ok = abs((subtotal + tax) - total) <= 0.05
        checks["subtotal_plus_tax_equals_total"] = sum_ok
        if not sum_ok:
            issues.append("Subtotal + Tax != Total (±0.05)")

    # VAT ~ 15% check
    if subtotal is not None and tax is not None:
        vat_ok = abs(tax - (subtotal * VAT_KSA)) <= max(0.5, 0.02 * subtotal)
        checks["vat_rate_ok"] = vat_ok
        if not vat_ok:
            issues.append("VAT not ~15% of Subtotal")

    # Venue profile matching (optional: use keywords/hints if your store has them)
    vm = match_venue_profile(data)
    checks["venue_match"] = vm
    if vm.get("matched") is False:
        # not necessarily an issue, but reduces confidence
        pass

    # Score heuristics (editable)
    fraud = 0
    confidence = 30
    if checks.get("subtotal_plus_tax_equals_total"): confidence += 30
    if checks.get("vat_rate_ok"): confidence += 20
    if vm.get("matched"): confidence += 20

    if not checks.get("subtotal_plus_tax_equals_total"): fraud += 30
    if not checks.get("vat_rate_ok"): fraud += 20

    # Basic clamp
    fraud = max(0, min(100, fraud))
    confidence = max(0, min(100, confidence))

    return {
        "fraudScore": fraud,
        "confidenceScore": confidence,
        "checks": checks,
        "issues": issues,
    }