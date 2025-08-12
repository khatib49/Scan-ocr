import json, re, unicodedata
from typing import List, Dict, Optional, Union
from rapidfuzz import fuzz

ARABIC_MAP = {
    "أ": "ا", "إ": "ا", "آ": "ا",
    "ة": "ه",
    "ى": "ي",
    "ؤ": "و", "ئ": "ي",
}

def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\u0600-\u06FF]+", " ", s)
    for k, v in ARABIC_MAP.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()

def _score(candidates: Union[str, List[str], None], query: str) -> int:
    if not candidates:
        return 0
    if isinstance(candidates, list):
        return max((fuzz.partial_ratio(normalize_text(c), normalize_text(query))
                   for c in candidates if isinstance(c, str)), default=0)
    return fuzz.partial_ratio(normalize_text(str(candidates)), normalize_text(query))

def find_best_profile(profiles: List[Dict], merchant: str, address: str) -> Optional[Dict]:
    m = normalize_text(merchant)
    a = normalize_text(address)
    if not (m or a):
        return None

    best = None
    best_total = 0.0
    for p in profiles:
        mk = p.get("MerchantName_Keyword")
        ak = p.get("MerchantAddress_Keyword")
        ms = _score(mk, m)
        as_ = _score(ak, a)
        total = ms * 0.7 + as_ * 0.3
        if total > best_total:
            best_total = total
            best = p

    return best if best_total >= 55 else None

def load_profiles(json_path: str) -> List[Dict[str, Any]]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []
