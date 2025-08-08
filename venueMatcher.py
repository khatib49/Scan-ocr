import json, re, unicodedata
from typing import List, Dict, Optional
from rapidfuzz import fuzz, process

def load_profiles(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

ARABIC_MAP = {
    "أ": "ا", "إ": "ا", "آ": "ا",
    "ة": "ه",
    "ى": "ي",
    "ؤ": "و", "ئ": "ي",
}

def normalize_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    # replace punctuation & extra spaces
    s = re.sub(r"[^\w\u0600-\u06FF]+", " ", s)
    # arabic normalization
    for k, v in ARABIC_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def best_score(candidates, query):
    # candidates can be str or list[str]
    if candidates is None:
        return 0
    if isinstance(candidates, list):
        return max((fuzz.partial_ratio(normalize_text(str(c)), normalize_text(query)) for c in candidates if isinstance(c, str)), default=0)
    return fuzz.partial_ratio(normalize_text(str(candidates)), normalize_text(query))

def find_best_profile(profiles: List[Dict], merchant: str, address: str) -> Optional[Dict]:
    m = normalize_text(merchant)
    a = normalize_text(address)
    if not (m or a):
        return None

    best = None
    best_total = 0
    for p in profiles:
        mk = p.get("MerchantName_Keyword")
        ak = p.get("MerchantAddress_Keyword")
        ms = best_score(mk, m)
        as_ = best_score(ak, a)
        total = ms * 0.7 + as_ * 0.3
        if total > best_total:
            best_total = total
            best = p

    # threshold to avoid wrong matches
    return best if best_total >= 55 else None
