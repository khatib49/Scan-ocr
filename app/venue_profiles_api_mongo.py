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
    from app import main as app_main
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



async def find_similar_profile(merchant_guess: str) -> Dict[str, Any]:
    """
    Finds the single best match in VenueProfile by name similarity and returns it
    ONLY if the best similarity >= 0.80. Otherwise returns matched=False.

    Scoring:
      - For each doc, gather candidate name strings:
          * profile.MerchantName (or MerchantName)
          * each item in profile.MerchantName_Keyword or MerchantName_Keyword
      - Score = max( fuzzy_ratio(guess, candidate_name) )  # 0..1
      - Choose the document with the highest score.
    """

    out: Dict[str, Any] = {"matched": False, "profile": None, "signals": {}, "candidates": []}

    guess_raw = (merchant_guess or "").strip()
    if not guess_raw:
        return out

    # Early token sanity (optional, just to avoid garbage inputs)
    guess_tokens = tokenize_distinct(guess_raw)
    if not guess_tokens:
        return out

    # --- 1) Pull a reasonable top-K candidate set from Mongo, using text index first
    K = 20
    docs = []
    cur = (DB["VenueProfile"]
           .find({"$text": {"$search": guess_raw}}, {"score": {"$meta": "textScore"}})
           .sort([("score", {"$meta": "textScore"})])
           .limit(K))
    docs = await cur.to_list(length=K)

    # Fallback: regex on a few tokens if text index returns nothing
    if not docs:
        tokens = [t for t in re.split(r"\s+", guess_raw) if len(t) > 2][:3]
        if tokens:
            regex = "|".join(map(re.escape, tokens))
            cur = (DB["VenueProfile"]
                   .find({
                       "$or": [
                           {"profile.MerchantName": {"$regex": regex, "$options": "i"}},
                           {"MerchantName": {"$regex": regex, "$options": "i"}},
                           {"profile.MerchantName_Keyword": {"$regex": regex, "$options": "i"}},
                           {"MerchantName_Keyword": {"$regex": regex, "$options": "i"}},
                       ]
                   })
                   .limit(K))
            docs = await cur.to_list(length=K)

    if not docs:
        return out

    # --- 2) For each doc, compute the best fuzzy similarity vs any candidate name string
    def get_profile(doc: Dict[str, Any]) -> Dict[str, Any]:
        return doc.get("profile") if isinstance(doc.get("profile"), dict) else doc

    def get_candidate_strings(doc: Dict[str, Any]) -> List[str]:
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
                    # If profile missing, try root doc
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

    scored: List[Tuple[Dict[str, Any], float, str]] = []  # (doc, best_sim, best_name_used)

    for doc in docs:
        candidates = get_candidate_strings(doc)
        if not candidates:
            continue

        best_sim = 0.0
        best_name = ""
        for cand in candidates:
            sim = fuzzy_ratio(guess_raw, cand)  # expected 0..1
            if sim > best_sim:
                best_sim = sim
                best_name = cand

        scored.append((doc, best_sim, best_name))

        prof = get_profile(doc)
        out["candidates"].append({
            "id": str(doc.get("_id")),
            "name": best_name or prof.get("MerchantName") or "",
            "final": best_sim
        })

    if not scored:
        return out

    # --- 3) Select the absolute best and apply a single 80% threshold gate
    scored.sort(key=lambda x: x[1], reverse=True)
    best_doc, best_sim, best_name = scored[0]
    second_sim = scored[1][1] if len(scored) > 1 else 0.0

    THRESHOLD = 0.80  # 80%
    if best_sim < THRESHOLD:
        out["matched"] = False
        out["profile"] = None
        out["signals"] = {
            "best_similarity": best_sim,
            "second_similarity": second_sim,
            "threshold": THRESHOLD,
            "best_name": best_name
        }
        return out

    # Success
    prof = get_profile(best_doc)
    out["matched"] = True
    out["profile"] = prof
    out["signals"] = {
        "best_similarity": best_sim,
        "second_similarity": second_sim,
        "threshold": THRESHOLD,
        "best_name": best_name
    }
    return out

