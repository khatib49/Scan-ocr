import re
from difflib import SequenceMatcher
from typing import List, Set
import unicodedata

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
    "Boulevard", "Promenade", "City",
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
    s = _strip_latin_diacritics(s)  
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

def _strip_latin_diacritics(text: str) -> str:
    """
    Turn 'HÖCHÖ' -> 'HOCHO' by removing combining marks (NFKD).
    Keeps Arabic letters intact.
    """
    if not text:
        return ""
    # NFKD: expand Ö => 'O' + combining diaeresis; then drop combining marks
    nk = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nk if not unicodedata.combining(ch))

def tokenize_distinct(s: str) -> list[str]:
    s_norm = arabic_normalize(s).strip()
    s_lower = s_norm.lower()

    # --- special case: don't remove generic for pure "location coffee" ---
    if s_lower.replace(",", "").replace("-", "").strip() in {"location coffee"}:
        toks = [t for t in re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", s_lower) if len(t) >= 3]
        seen, out = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t); out.append(t)
        return out

    # 1) normal tokenization
    toks = re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", s_lower)

    # 2) length filter (default >=3)
    toks = [t for t in toks if len(t) >= 3]

    # 3) remove generics if anything remains
    non_generic = [t for t in toks if t not in GENERIC_WORDS]
    if non_generic:
        toks = non_generic

    # 4) Fallbacks to avoid "all generic/empty" for short brand names like HÖCHÖ:
    if not toks:
        # try allowing 2-char tokens (brands with short chunks)
        toks2 = re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", s_lower)
        toks2 = [t for t in toks2 if len(t) >= 2 and t not in GENERIC_WORDS]
        if toks2:
            toks = toks2

    # 5) Final fallback: if still empty but we have letters, keep the whole normalized word
    if not toks and s_lower:
        compact = re.sub(r"[^A-Za-z0-9\u0600-\u06FF]+", "", s_lower)
        if len(compact) >= 2:
            toks = [compact]

    # de-dupe, keep order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
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
