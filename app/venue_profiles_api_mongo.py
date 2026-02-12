# app/venue_profiles_api.py  (Mongo-backed)
import os, json, asyncio, hashlib
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Body, Security, Query
from bson import ObjectId

# Reuse your existing auth + DB handle style (like in projects.py)
from app.security import verify_admin_key, _mongo_db as DB
from app.venue_matcher import build_name_index
from app.text_utils import fuzzy_ratio, jaccard,tokenize_distinct

router = APIRouter(prefix="/venue-profiles-mongo", tags=["Venue Profiles (Mongo)"])

_profiles_lock = asyncio.Lock()
COLL = lambda: DB["VenueProfile"]

# ---------- helpers ----------
def _norm_id(mid: Any) -> str:
    return "" if mid is None else str(mid)

def _ensure_keywords_list(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize MerchantName_Keyword & MerchantAddress_Keyword to always be lists of strings.
    """
    def to_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            return [s.strip() for s in [v] if s.strip()]
        return []
    p = dict(p)
    p["MerchantName_Keyword"] = to_list(p.get("MerchantName_Keyword"))
    p["MerchantAddress_Keyword"] = to_list(p.get("MerchantAddress_Keyword"))
    return p

async def _load_all_from_db() -> List[Dict[str, Any]]:
    cur = COLL().find({})
    out = []
    async for d in cur:
        # make Mongo documents JSON-serializable (drop _id)
        if "_id" in d and isinstance(d["_id"], ObjectId):
            d["_mongo_id"] = str(d["_id"])
            del d["_id"]
        out.append(d)
    return out

async def _save_cache(profiles: List[Dict[str, Any]]) -> None:
    # Validate index builds before hot-swapping memory for /analyze
    build_name_index(profiles)
    from app import main_openai as app_main
    app_main.VENUE_PROFILES = profiles
    app_main.NAME_INDEX = build_name_index(profiles)

async def _reload_cache_from_db() -> Dict[str, Any]:
    profiles = await _load_all_from_db()
    await _save_cache(profiles)
    return {"ok": True, "reloaded": True, "count": len(profiles)}

# ---------- endpoints ----------

@router.get(
    "/all",
    dependencies=[Security(verify_admin_key)],
    summary="Get all venue profiles (admin)",
)
async def get_all_profiles(
    q: Optional[str] = Query(None, description="Case-insensitive search in MerchantName_Keyword or MerchantAddress_Keyword"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    flt: Dict[str, Any] = {}
    if q:
        flt["$or"] = [
            {"MerchantName_Keyword": {"$regex": q, "$options": "i"}},
            {"MerchantAddress_Keyword": {"$regex": q, "$options": "i"}},
        ]
    cur = COLL().find(flt).skip(offset).limit(limit)
    docs: List[Dict[str, Any]] = []
    async for d in cur:
        d["id"] = str(d["_id"])
        del d["_id"]
        docs.append(d)
    count = await COLL().count_documents(flt)
    return {"count": count, "profiles": docs}

@router.get(
    "/meta",
    dependencies=[Security(verify_admin_key)],
    summary="Get metadata (admin)",
)
async def get_meta() -> Dict[str, Any]:
    count = await COLL().estimated_document_count()
    # Optionally: keep a small hash of current export for sanity
    sample = await COLL().find({}).limit(100).to_list(length=100)
    blob = json.dumps(sample, ensure_ascii=False, default=str).encode("utf-8")
    sha256 = hashlib.sha256(blob).hexdigest()
    return {
        "meta": {"collection": "VenueProfile", "db": DB.name, "estimated_count": count},
        "count": count,
        "sha256_sample100": sha256,
    }

@router.post(
    "/reload",
    dependencies=[Security(verify_admin_key)],
    summary="Reload cache from Mongo into memory (admin)",
)
async def reload_from_db() -> Dict[str, Any]:
    async with _profiles_lock:
        return await _reload_cache_from_db()

@router.get(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Get a merchant profile (admin)",
)
async def get_merchant(merchant_id: str) -> Dict[str, Any]:
    # Match either numeric or string MerchantId
    cand = [{"MerchantId": merchant_id}]
    try:
        cand.append({"MerchantId": int(merchant_id)})
    except Exception:
        pass
    doc = await COLL().find_one({"$or": cand})
    if not doc:
        raise HTTPException(404, detail="Merchant not found")
    doc.pop("_id", None)
    doc.pop("id", None)
    doc.pop("CreatedAt", None)
    doc.pop("_seed", None)
    return {"profile": doc}

@router.post(
    "",
    dependencies=[Security(verify_admin_key)],
    summary="Add a merchant profile (admin)",
)
async def add_merchant(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    normalized = _ensure_keywords_list(payload)
    if "MerchantId" not in normalized:
        raise HTTPException(400, detail="Missing required field: MerchantId")
    # Enforce uniqueness on MerchantId if present
    mid = normalized.get("MerchantId")
    exists = await COLL().find_one({"MerchantId": {"$in": [mid, _norm_id(mid)]}})
    if exists:
        raise HTTPException(409, detail="MerchantId already exists")

    normalized["CreatedAt"] = datetime.now(timezone.utc)
    res = await COLL().insert_one(normalized)
    created = await COLL().find_one({"_id": res.inserted_id})
    created["id"] = str(created["_id"]); del created["_id"]

    # hot-reload cache
    async with _profiles_lock:
        await _reload_cache_from_db()

    return {"ok": True, "created": True, "profile": created}

@router.patch(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Update a merchant profile by MerchantId (admin)",
)
async def update_merchant(
    merchant_id: str,
    patch: Dict[str, Any] = Body(..., description="Fields to update"),
) -> Dict[str, Any]:
    if "MerchantId" in patch and _norm_id(patch["MerchantId"]) != _norm_id(merchant_id):
        raise HTTPException(400, detail="Cannot change MerchantId; use the id in the path")

    normalized = _ensure_keywords_list(patch)

    # Find existing by MerchantId (string or int)
    cand = [{"MerchantId": merchant_id}]
    try:
        cand.append({"MerchantId": int(merchant_id)})
    except Exception:
        pass

    doc = await COLL().find_one_and_update(
        {"$or": cand},
        {"$set": normalized, "$setOnInsert": {"UpdatedAt": datetime.now(timezone.utc)}},
        return_document=True,
    )
    if not doc:
        raise HTTPException(404, detail="Merchant not found")
    doc["id"] = str(doc["_id"]); del doc["_id"]

    async with _profiles_lock:
        await _reload_cache_from_db()

    return {"ok": True, "updated": True, "profile": doc}

@router.delete(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Delete a merchant profile by MerchantId (admin)",
)
async def delete_merchant(merchant_id: str) -> Dict[str, Any]:
    cand = [{"MerchantId": merchant_id}]
    try:
        cand.append({"MerchantId": int(merchant_id)})
    except Exception:
        pass

    doc = await COLL().find_one({"$or": cand})
    if not doc:
        raise HTTPException(404, detail="Merchant not found")

    await COLL().delete_one({"_id": doc["_id"]})

    async with _profiles_lock:
        await _reload_cache_from_db()

    doc["id"] = str(doc["_id"]); del doc["_id"]
    return {"ok": True, "deleted": True, "removed": doc}



async def find_similar_profile(merchant_guess: str, address_guess: str = None) -> Dict[str, Any]:
    """
    Finds the single best match in VenueProfile using flexible matching modes.
    
    MATCHING MODES:
    
    MODE 1: STRICT (Both name AND address available)
      - Receipt has address AND profile has address
      - Name similarity must be >= 75%
      - Address similarity must be >= 70%
      - BOTH thresholds must pass
      
    MODE 2: NAME-ONLY (No address on receipt)
      - Receipt has NO address
      - Profile also has NO address
      - Name similarity must be >= 85%
      - Requires strong token matching
    
    Returns:
      {
        "matched": bool,
        "profile": dict or None,
        "signals": {
          "name_fuzzy": float,
          "address_fuzzy": float or None,
          "contains_strong_token": bool,
          "match_mode": "strict" or "name_only",
          "rejection_reason": str or None,
          "best_name": str,
          "best_address": str or None,
          "thresholds": {...}
        },
        "candidates": [...]
      }
    """

    out: Dict[str, Any] = {"matched": False, "profile": None, "signals": {}, "candidates": []}

    guess_raw = (merchant_guess or "").strip()
    if not guess_raw:
        return out

    address_raw = (address_guess or "").strip() if address_guess else None

    # Early token sanity (optional, just to avoid garbage inputs)
    guess_tokens = tokenize_distinct(guess_raw)
    if not guess_tokens:
        return out

    # --- CONFIGURATION ---
    # FLEXIBLE MATCHING: Support both strict and name-only modes
    
    # MODE 1: STRICT (when both receipt and profile have addresses)
    NAME_MIN_THRESHOLD_STRICT = 0.75       # Name must be at least 75% similar
    ADDRESS_MIN_THRESHOLD = 0.70           # Address must be at least 70% similar
    
    # MODE 2: NAME-ONLY (when receipt has NO address)
    NAME_MIN_THRESHOLD_NAME_ONLY = 0.85    # Name must be at least 85% similar
    # Requires strong token matching (no address to verify)
    
    # --- 1) Pull a reasonable top-K candidate set from Mongo, using text index first
    K = 20
    docs = []
    
    # Build search query - include address in text search if provided
    search_text = guess_raw
    if address_raw:
        search_text = f"{guess_raw} {address_raw}"
    
    cur = (DB["VenueProfile"]
           .find({"$text": {"$search": search_text}}, {"score": {"$meta": "textScore"}})
           .sort([("score", {"$meta": "textScore"})])
           .limit(K))
    docs = await cur.to_list(length=K)

    # Fallback: regex on a few tokens if text index returns nothing
    if not docs:
        tokens = [t for t in re.split(r"\s+", guess_raw) if len(t) > 2][:3]
        
        # Add address tokens if available
        if address_raw:
            addr_tokens = [t for t in re.split(r"\s+", address_raw) if len(t) > 2][:2]
            tokens.extend(addr_tokens)
        
        if tokens:
            regex = "|".join(map(re.escape, tokens))
            cur = (DB["VenueProfile"]
                   .find({
                       "$or": [
                           {"profile.MerchantName": {"$regex": regex, "$options": "i"}},
                           {"MerchantName": {"$regex": regex, "$options": "i"}},
                           {"profile.MerchantName_Keyword": {"$regex": regex, "$options": "i"}},
                           {"MerchantName_Keyword": {"$regex": regex, "$options": "i"}},
                           {"profile.MerchantAddress": {"$regex": regex, "$options": "i"}},
                           {"MerchantAddress": {"$regex": regex, "$options": "i"}},
                       ]
                   })
                   .limit(K))
            docs = await cur.to_list(length=K)

    if not docs:
        return out

    # --- 2) Helper functions to extract candidate strings ---
    
    def get_profile(doc: Dict[str, Any]) -> Dict[str, Any]:
        return doc.get("profile") if isinstance(doc.get("profile"), dict) else doc

    def get_candidate_names(doc: Dict[str, Any]) -> List[str]:
        """Extract all candidate name strings from a document"""
        prof = get_profile(doc)
        names: List[str] = []
        
        # Primary name
        if isinstance(prof.get("MerchantName"), str) and prof["MerchantName"].strip():
            names.append(prof["MerchantName"].strip())
        if isinstance(doc.get("MerchantName"), str) and doc["MerchantName"].strip():
            names.append(doc["MerchantName"].strip())

        # Keyword arrays (both root and nested)
        for path in (
            ("profile", "MerchantName_Keyword"),
            ("MerchantName_Keyword",),
        ):
            try:
                val = prof[path[1]] if len(path) == 1 else prof.get(path[1])
                if val is None and len(path) == 2 and path[0] == "profile":
                    val = doc.get(path[1])
                if isinstance(val, list):
                    for s in val:
                        if isinstance(s, str) and s.strip():
                            names.append(s.strip())
            except Exception:
                pass

        # Deduplicate while preserving order
        seen, uniq = set(), []
        for s in names:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    def get_candidate_addresses(doc: Dict[str, Any]) -> List[str]:
        """Extract all candidate address strings from a document"""
        prof = get_profile(doc)
        addresses: List[str] = []
        
        # Primary address
        if isinstance(prof.get("MerchantAddress"), str) and prof["MerchantAddress"].strip():
            addresses.append(prof["MerchantAddress"].strip())
        if isinstance(doc.get("MerchantAddress"), str) and doc["MerchantAddress"].strip():
            addresses.append(doc["MerchantAddress"].strip())
        
        # Address keyword arrays (if you have them)
        for path in (
            ("profile", "MerchantAddress_Keyword"),
            ("MerchantAddress_Keyword",),
        ):
            try:
                val = prof[path[1]] if len(path) == 1 else prof.get(path[1])
                if val is None and len(path) == 2 and path[0] == "profile":
                    val = doc.get(path[1])
                if isinstance(val, list):
                    for s in val:
                        if isinstance(s, str) and s.strip():
                            addresses.append(s.strip())
            except Exception:
                pass

        # Deduplicate while preserving order
        seen, uniq = set(), []
        for s in addresses:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    def check_strong_tokens(guess: str, candidate: str) -> bool:
        """Check if guess contains strong matching tokens (3+ chars)"""
        guess_tokens = set(t.lower() for t in re.split(r"\s+", guess) if len(t) >= 3)
        cand_tokens = set(t.lower() for t in re.split(r"\s+", candidate) if len(t) >= 3)
        
        if not guess_tokens or not cand_tokens:
            return False
        
        # At least 50% of guess tokens must appear in candidate
        matches = guess_tokens & cand_tokens
        return len(matches) >= max(1, len(guess_tokens) * 0.5)

    # --- 3) Score each document ---
    
    scored: List[Tuple[Dict[str, Any], float, float, float, str, str]] = []  
    # (doc, name_score, address_score, combined_score, best_name_used, best_address_used)

    for doc in docs:
        # Get all candidate strings
        candidate_names = get_candidate_names(doc)
        candidate_addresses = get_candidate_addresses(doc) if address_raw else []
        
        if not candidate_names:
            continue

        # Score name similarity
        best_name_sim = 0.0
        best_name = ""
        has_strong_tokens = False
        
        for cand_name in candidate_names:
            sim = fuzzy_ratio(guess_raw, cand_name)
            if sim > best_name_sim:
                best_name_sim = sim
                best_name = cand_name
                has_strong_tokens = check_strong_tokens(guess_raw, cand_name)

        # Score address similarity (if address provided)
        best_addr_sim = 0.0
        best_addr = ""
        
        if address_raw and candidate_addresses:
            for cand_addr in candidate_addresses:
                sim = fuzzy_ratio(address_raw, cand_addr)
                if sim > best_addr_sim:
                    best_addr_sim = sim
                    best_addr = cand_addr

        # Store both name and address scores separately
        # No weighted combination - both must meet their own thresholds

        scored.append((doc, best_name_sim, best_addr_sim, best_name, best_addr))

        # Store candidate info
        prof = get_profile(doc)
        out["candidates"].append({
            "id": str(doc.get("_id")),
            "name": best_name or prof.get("MerchantName") or "",
            "address": best_addr or prof.get("MerchantAddress") or "",
            "name_score": best_name_sim,
            "address_score": best_addr_sim
        })

    if not scored:
        return out

    # --- 4) Select the best match and apply appropriate thresholds based on mode ---
    
    # Sort by name score first (primary), then address score (secondary)
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_doc, name_score, addr_score, best_name, best_addr = scored[0]
    second_name_score = scored[1][1] if len(scored) > 1 else 0.0

    # Determine matching mode
    receipt_has_address = bool(address_raw and address_raw.strip())
    
    matched = False
    rejection_reason = None
    match_mode = None
    
    if receipt_has_address:
        # MODE 1: STRICT - Receipt has address, so we need address matching
        match_mode = "strict"
        
        # Get profile info to check if it has an address
        prof = get_profile(best_doc)
        profile_has_address = bool(
            (prof.get("MerchantAddress") and str(prof.get("MerchantAddress")).strip()) or
            (best_doc.get("MerchantAddress") and str(best_doc.get("MerchantAddress")).strip())
        )
        
        if not profile_has_address or addr_score == 0.0:
            # Receipt has address but profile doesn't - cannot match
            matched = False
            rejection_reason = "Receipt has address but profile has no address to compare"
        elif name_score >= NAME_MIN_THRESHOLD_STRICT and addr_score >= ADDRESS_MIN_THRESHOLD:
            # Both name and address meet thresholds
            matched = True
            rejection_reason = None
        else:
            # One or both scores too low
            matched = False
            if name_score < NAME_MIN_THRESHOLD_STRICT and addr_score < ADDRESS_MIN_THRESHOLD:
                rejection_reason = f"Both name ({name_score:.2f} < {NAME_MIN_THRESHOLD_STRICT}) and address ({addr_score:.2f} < {ADDRESS_MIN_THRESHOLD}) scores too low"
            elif name_score < NAME_MIN_THRESHOLD_STRICT:
                rejection_reason = f"Name score too low ({name_score:.2f} < {NAME_MIN_THRESHOLD_STRICT})"
            else:
                rejection_reason = f"Address score too low ({addr_score:.2f} < {ADDRESS_MIN_THRESHOLD})"
    
    else:
        # MODE 2: NAME-ONLY - Receipt has NO address
        match_mode = "name_only"
        
        # Get profile info to check if it ALSO has no address
        prof = get_profile(best_doc)
        profile_has_address = bool(
            (prof.get("MerchantAddress") and str(prof.get("MerchantAddress")).strip()) or
            (best_doc.get("MerchantAddress") and str(best_doc.get("MerchantAddress")).strip())
        )
        
        if profile_has_address:
            # Receipt has no address but profile has address - cannot match
            # (We can only match address-less to address-less)
            matched = False
            rejection_reason = "Receipt has no address but profile requires address verification"
        elif name_score >= NAME_MIN_THRESHOLD_NAME_ONLY and has_strong_tokens:
            # High name match with strong tokens
            matched = True
            rejection_reason = None
        else:
            # Name score too low or no strong tokens
            matched = False
            if name_score < NAME_MIN_THRESHOLD_NAME_ONLY:
                rejection_reason = f"Name score too low for address-less matching ({name_score:.2f} < {NAME_MIN_THRESHOLD_NAME_ONLY})"
            else:
                rejection_reason = f"No strong token matching for address-less receipt"

    # Check for strong token match
    has_strong_tokens = check_strong_tokens(guess_raw, best_name)

    # Build signals
    out["signals"] = {
        "name_fuzzy": name_score,
        "address_fuzzy": addr_score if address_raw else None,
        "second_best_name_score": second_name_score,
        "contains_strong_token": has_strong_tokens,
        "match_mode": match_mode,  # "strict" or "name_only"
        "best_name": best_name,
        "best_address": best_addr if address_raw else None,
        "rejection_reason": rejection_reason,
        "thresholds": {
            "strict": {
                "name_min": NAME_MIN_THRESHOLD_STRICT,
                "address_min": ADDRESS_MIN_THRESHOLD
            },
            "name_only": {
                "name_min": NAME_MIN_THRESHOLD_NAME_ONLY,
                "requires_strong_tokens": True
            }
        }
    }

    if matched:
        prof = get_profile(best_doc)
        out["matched"] = True
        out["profile"] = prof
    else:
        out["matched"] = False
        out["profile"] = None

    return out


# --- HELPER FUNCTIONS (you may already have these) ---

def tokenize_distinct(text: str) -> List[str]:
    """Tokenize text into distinct non-empty tokens"""
    tokens = re.split(r"\s+", text.strip())
    seen = set()
    result = []
    for t in tokens:
        t_lower = t.lower()
        if t_lower and t_lower not in seen:
            seen.add(t_lower)
            result.append(t)
    return result


def fuzzy_ratio(s1: str, s2: str) -> float:
    """
    Calculate fuzzy similarity ratio between two strings.
    Returns a value between 0.0 and 1.0
    
    You can use fuzzywuzzy or rapidfuzz library:
    from rapidfuzz import fuzz
    return fuzz.ratio(s1.lower(), s2.lower()) / 100.0
    
    Or implement your own logic
    """
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(s1.lower(), s2.lower()) / 100.0
    except ImportError:
        # Fallback to simple character-based similarity
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        
        if s1_lower == s2_lower:
            return 1.0
        
        # Simple character overlap ratio
        set1 = set(s1_lower)
        set2 = set(s2_lower)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    