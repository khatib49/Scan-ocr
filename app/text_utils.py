import re
from difflib import SequenceMatcher
from typing import List, Set

_AR_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]")  # fatha/damma/.. + superscript alef
_PUNCT_SPACE_RE   = re.compile(r"[^0-9A-Za-z\u0600-\u06FF]+")  # keep Arabic letters
_TATWEEL_RE       = re.compile(r"[\u0640]")

# Generic / geo / business / venue words — ignored in similarity comparison
GENERIC_WORDS = {
    # --- Arabic venue/common ---
    "كافيه", "كوفي", "مقهى", "مطعم", "متجر", "سوق", "مارت", "فود",
    # --- Arabic cities / areas ---
    "جدة", "جده", "الرياض", "بوليفارد", "مارينا", "ممشى", "بروميناد",
    "برومينادز", "ڤيا", "فيا", "البلد", "كورنيش",
    # --- Arabic companies / suffixes ---
    "شركة", "شركه", "مؤسسة", "للتجارة", "للتجاره", "للتوزيع", "للمقاولات",
    "محدودة", "المحدودة", "ذ م م", "ذ.م.م", "فرع", "فرعاً",
    # --- English venue/common ---
    "cafe", "coffee", "restaurant", "shop", "store", "market", "mart",
    "food", "lounge", "mall", "center", "centre", "co", "company",
    # --- English cities / landmarks ---
    "riyadh", "jeddah", "blvd", "boulevard", "marina", "promenade",
    "via", "jyc", "yacht", "club", "yachtclub", "jewelry", "jewelery",
    # --- Business suffixes / forms ---
    "est", "est.", "trading", "tradingco", "trading-co", "establishment",
    "llc", "inc", "limited", "enterprise", "group",
    # --- Composite phrases (treated tokenwise) ---
    "yacht club", "blvd city", "trading co", "trading company",
    # --- Misc fillers often seen ---
    "ksa", "saudi", "arabia", "branch", "store", "location", "place"
}


def arabic_normalize(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    # Unify common Arabic forms
    s = (s.replace("أ", "ا")
           .replace("إ", "ا")
           .replace("آ", "ا")
           .replace("ى", "ي")
           .replace("ة", "ه"))
    # Remove diacritics and tatweel
    s = _AR_DIACRITICS_RE.sub("", s)
    s = _TATWEEL_RE.sub("", s)
    # Collapse punctuation/extra spaces
    s = _PUNCT_SPACE_RE.sub(" ", s)
    return " ".join(s.split()).lower()

def tokenize_distinct(s: str) -> List[str]:
    s = arabic_normalize(s)
    toks = [t for t in s.split() if len(t) >= 3 and t not in GENERIC_WORDS]
    # keep order but make distinct
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))

def fuzzy_ratio(a: str, b: str) -> float:
    """0..1 similarity using difflib; works without extra deps."""
    if not a or not b:
        return 0.0
    # Compare normalized strings (remove spaces to be less sensitive)
    an = arabic_normalize(a).replace(" ", "")
    bn = arabic_normalize(b).replace(" ", "")
    return SequenceMatcher(None, an, bn).ratio()
