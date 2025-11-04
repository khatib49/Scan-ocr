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
    Mongo-powered fuzzy finder with re-ranking.
    Returns:
      {
        "matched": bool,
        "profile": dict | None,
        "signals": {
            "top_textScore": float,
            "final": float,               # our combined score 0..1
            "overlap": float,             # Jaccard 0..1
            "name_fuzzy": float,          # 0..1 (difflib)
            "contains_strong_token": bool,
            "second_final": float         # for margin diagnostics
        },
        "candidates": [ { "id": <_id>, "textScore": x, "final": y, "name": "..."} ]  # (optional, for logs)
      }
    """
    out: Dict[str, Any] = {"matched": False, "profile": None, "signals": {}, "candidates": []}

    guess_raw = (merchant_guess or "").strip()
    if not guess_raw:
        return out

    guess_tokens = set(tokenize_distinct(guess_raw))
    if not guess_tokens:
        # If all tokens were generic, don't attempt a match
        return out

    # ---- 1) Pull top-K by Mongo textScore
    K = 10
    cur = (DB["VenueProfile"]
            .find({"$text": {"$search": guess_raw}},
                  {"score": {"$meta": "textScore"}})
            .sort([("score", {"$meta": "textScore"})])
            .limit(K))

    docs = await cur.to_list(K)
    if not docs:
        # Fallback regex on name keywords (still re-ranked)
        tokens = [t for t in re.split(r"\s+", guess_raw) if len(t) > 2][:3]
        if tokens:
            regex = "|".join(map(re.escape, tokens))
            cur = (DB["VenueProfile"]
                    .find({"$or": [
                        {"profile.MerchantName_Keyword": {"$regex": regex, "$options": "i"}},
                        {"MerchantName_Keyword": {"$regex": regex, "$options": "i"}},
                    ]})
                    .limit(K))
            docs = await cur.to_list(K)

    if not docs:
        return out

    # ---- 2) Build candidates with signals
    def extract_keywords(doc: Dict[str, Any]) -> List[str]:
        prof = doc.get("profile") if isinstance(doc.get("profile"), dict) else doc
        kw = prof.get("MerchantName_Keyword") or prof.get("profile", {}).get("MerchantName_Keyword")
        if not isinstance(kw, list):
            kw = []
        # Remove generic words-only entries
        cleaned = []
        for k in kw:
            k = (k or "").strip()
            if not k:
                continue
            toks = set(tokenize_distinct(k))
            if not toks:
                continue
            cleaned.append(k)
        return cleaned

    cand_list: List[Tuple[Dict[str, Any], float, float, float, bool]] = []
    # (doc, textScore, overlap, name_fuzzy, contains_strong_token)

    max_text = max((doc.get("score", 0.0) for doc in docs), default=0.0) or 1.0

    for doc in docs:
        prof = doc.get("profile") if isinstance(doc.get("profile"), dict) else doc
        text_score = float(doc.get("score", 0.0))
        kw_list = extract_keywords(doc)

        # Aggregate signals across keywords
        max_fuzzy = 0.0
        best_kw = ""
        any_strong_token = False
        # token overlap: compare set vs each keyword's tokens and keep the best
        max_overlap = 0.0

        for kw in kw_list:
            kw_tokens = set(tokenize_distinct(kw))
            if not kw_tokens:
                continue
            ov = jaccard(guess_tokens, kw_tokens)
            if ov > max_overlap:
                max_overlap = ov

            f = fuzzy_ratio(guess_raw, kw)
            if f > max_fuzzy:
                max_fuzzy = f
                best_kw = kw

            # strong token = at least one guess token appears verbatim in kw tokens
            if not any_strong_token and (guess_tokens & kw_tokens):
                any_strong_token = True

        cand_list.append((doc, text_score, max_overlap, max_fuzzy, any_strong_token))

        out["candidates"].append({
            "id": str(doc.get("_id")),
            "name": (prof.get("MerchantName") or best_kw or ""),
            "textScore": text_score,
            "overlap": max_overlap,
            "name_fuzzy": max_fuzzy
        })

    # ---- 3) Final combined score & selection
    # weights (tune on your data)
    W_TEXT, W_OVER, W_FUZZ = 0.50, 0.30, 0.20
    MIN_FINAL = 0.55
    MIN_FUZZY = 0.80
    MARGIN   = 0.15

    scored: List[Tuple[Dict[str, Any], float, Dict[str, Any]]] = []
    for doc, text_score, overlap, name_fuzzy, has_token in cand_list:
        final = (W_TEXT * (text_score / max_text)) + (W_OVER * overlap) + (W_FUZZ * name_fuzzy)
        signals = {
            "textScore_norm": (text_score / max_text),
            "overlap": overlap,
            "name_fuzzy": name_fuzzy,
            "contains_strong_token": bool(has_token),
            "textScore_raw": text_score,
            "final": final,
        }
        scored.append((doc, final, signals))

    # order by our final score
    scored.sort(key=lambda x: x[1], reverse=True)
    best_doc, best_final, best_signals = scored[0]
    second_final = scored[1][1] if len(scored) > 1 else 0.0

    # gate: need threshold + margin + at least one strong token + min fuzzy
    strong_enough = (
        (best_final >= MIN_FINAL) and
        ((best_final - second_final) >= MARGIN or best_final == 1.0) and
        best_signals["contains_strong_token"] and
        (best_signals["name_fuzzy"] >= MIN_FUZZY)
    )

    if not strong_enough:
        # no reliable match
        out["matched"] = False
        out["profile"] = None
        out["signals"] = {
            "top_textScore": scored[0][2]["textScore_raw"],
            "final": best_final,
            "overlap": best_signals["overlap"],
            "name_fuzzy": best_signals["name_fuzzy"],
            "contains_strong_token": best_signals["contains_strong_token"],
            "second_final": second_final
        }
        return out

    # Success
    prof = best_doc.get("profile") if isinstance(best_doc.get("profile"), dict) else best_doc
    out["matched"] = True
    out["profile"] = prof
    out["signals"] = {
        "top_textScore": scored[0][2]["textScore_raw"],
        "final": best_final,
        "overlap": best_signals["overlap"],
        "name_fuzzy": best_signals["name_fuzzy"],
        "contains_strong_token": best_signals["contains_strong_token"],
        "second_final": second_final
    }
    return out

