# venue_matcher.py
import re, unicodedata, json
from typing import Dict, Any, List, Optional
from pymongo import errors

def _strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def _normalize(s: Optional[str]) -> str:
    if not s:
        return ""
    s = _strip_diacritics(s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _names_from_profile(p: Dict[str, Any]) -> List[str]:
    # support multiple fields / aliases
    raw = []
    for k in ("MerchantName_Keyword", "TenantName", "Brand", "Aliases"):
        v = p.get(k)
        if isinstance(v, str):
            raw.extend([t.strip() for t in re.split(r"[|,/]", v) if t.strip()])
        elif isinstance(v, list):
            raw.extend([str(t).strip() for t in v if str(t).strip()])
    return list({r for r in raw if r})

def build_name_index(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for p in profiles:
        for n in _names_from_profile(p):
            nn = _normalize(n)
            if nn:
                idx[nn] = p  # last one wins if duplicates
    return idx

def load_profiles(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_best_profile_indexed(
    name_index: Dict[str, Dict[str, Any]],
    merchant_guess: Optional[str]
) -> Dict[str, Any]:
    ng = _normalize(merchant_guess)
    if not ng:
        return {"matched": False, "profile": None, "hints": {}}
    p = name_index.get(ng)
    if p:
        return {"matched": True, "profile": p, "hints": p.get("ExtractionHints", {})}
    return {"matched": False, "profile": None, "hints": {}}



async def ensure_venue_indexes(db):
    coll = db["VenueProfile"]
    info = await coll.index_information()
    # find any existing text index
    existing_text = next((n for n,s in info.items() if any(k[1] == 'text' for k in s.get('key',[]))), None)

    desired_keys = [
        ("profile.MerchantName_Keyword","text"),
        ("profile.TenantName","text"),
        ("profile.Brand","text"),
        ("profile.Aliases","text"),
    ]

    def keys_match(spec):
        k = spec.get("key", [])
        return len(k) == len(desired_keys) and all(a==b for a,b in zip(k, desired_keys))

    if existing_text and not keys_match(info[existing_text]):
        try:
            await coll.drop_index(existing_text)
        except errors.OperationFailure:
            pass
        existing_text = None

    if not existing_text:
        await coll.create_index(
            desired_keys,
            name="venue_text_idx",
            default_language="none",
            language_override="none",
            weights={
                "profile.MerchantName_Keyword": 10,
                "profile.Brand": 8,
                "profile.TenantName": 5,
                "profile.Aliases": 5
            },
        )

