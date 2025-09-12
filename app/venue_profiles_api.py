# app/venue_profiles_api.py
import os, json, hashlib, asyncio, tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Body, Query, Security
from app.security import verify_admin_key                 # admin header: X-Admin-Key
from app.venue_matcher import build_name_index            # reuse your indexer

router = APIRouter(prefix="/venue-profiles", tags=["Venue Profiles"])

# Single lock to avoid concurrent writes/reads during a replace
_profiles_lock = asyncio.Lock()

def _profiles_path() -> str:
    return os.getenv("VENUE_PROFILES_PATH", "data/venue_profiles.json")

def _validate_profiles(profiles: Any) -> List[Dict[str, Any]]:
    if not isinstance(profiles, list):
        raise HTTPException(400, detail="Payload must be a JSON array of profiles")

    ids = set()
    for i, p in enumerate(profiles):
        if not isinstance(p, dict):
            raise HTTPException(400, detail=f"Profile #{i} must be an object")

        # Minimal fields your matcher relies on
        mid = p.get("MerchantId")
        mnk = p.get("MerchantName_Keyword")
        mak = p.get("MerchantAddress_Keyword")

        if mid is None or not isinstance(mid, (int, str)):
            raise HTTPException(400, detail=f"Profile #{i} missing/invalid MerchantId")
        if not isinstance(mnk, list):
            raise HTTPException(400, detail=f"Profile #{i} missing/invalid MerchantName_Keyword[]")
        if not isinstance(mak, list):
            raise HTTPException(400, detail=f"Profile #{i} missing/invalid MerchantAddress_Keyword[]")

        if mid in ids:
            raise HTTPException(400, detail=f"Duplicate MerchantId: {mid}")
        ids.add(mid)

    # Make sure the index can be built
    build_name_index(profiles)
    return profiles

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _atomic_write(path: str, content: bytes) -> None:
    # Write to a temp file next to the target, then replace atomically
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as tf:
        tf.write(content)
        tmp = tf.name
    os.replace(tmp, path)

def _file_meta(path: str) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        return {
            "path": path,
            "exists": True,
            "size_bytes": st.st_size,
            "last_modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
        }
    except FileNotFoundError:
        return {"path": path, "exists": False}

@router.get(
    "",
    dependencies=[Security(verify_admin_key)],
    summary="Get current venue-profiles metadata (admin)",
)
async def get_meta() -> Dict[str, Any]:
    path = _profiles_path()
    meta = _file_meta(path)
    if not meta["exists"]:
        return {"meta": meta, "count": 0, "sha256": None}

    with open(path, "rb") as f:
        raw = f.read()
    try:
        data = json.loads(raw.decode("utf-8"))
        count = len(data) if isinstance(data, list) else 0
    except Exception:
        count = 0
    return {"meta": meta, "count": count, "sha256": _sha256_bytes(raw)}

@router.post(
    "/reload",
    dependencies=[Security(verify_admin_key)],
    summary="Reload venue-profiles from disk into memory (admin)",
)
async def reload_from_disk() -> Dict[str, Any]:
    # Lazy import to avoid circular import at module import time
    from app import main as app_main

    path = _profiles_path()
    if not os.path.exists(path):
        raise HTTPException(404, detail=f"File not found: {path}")

    async with _profiles_lock:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profiles = _validate_profiles(data)

        # Hot-swap in-memory state used by /analyze
        app_main.VENUE_PROFILES = profiles
        app_main.NAME_INDEX = build_name_index(profiles)

    return {"ok": True, "reloaded": True, "count": len(profiles)}

@router.put(
    "",
    dependencies=[Security(verify_admin_key)],
    summary="Replace venue-profiles with a new file or JSON body (admin)",
)
async def replace_profiles(
    # Option 1: multipart upload
    file: Optional[UploadFile] = File(None, description="JSON file to replace the profiles"),
    # Option 2: raw JSON body
    payload: Optional[List[Dict[str, Any]]] = Body(None, description="Profiles array"),
    dry_run: bool = Query(False, description="Validate only; don't write/replace"),
) -> Dict[str, Any]:
    # Accept either multipart file or JSON body
    if not file and payload is None:
        raise HTTPException(400, detail="Provide either 'file' or JSON body")

    if file:
        content = await file.read()
        try:
            parsed = json.loads(content.decode("utf-8"))
        except Exception as e:
            raise HTTPException(400, detail=f"Invalid JSON file: {e}")
    else:
        parsed = payload
        content = json.dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8")

    profiles = _validate_profiles(parsed)

    if dry_run:
        return {"ok": True, "dry_run": True, "count": len(profiles)}

    path = _profiles_path()

    # Lazy import to avoid circular import at import-time
    from app import main as app_main

    async with _profiles_lock:
        # 1) Persist atomically
        _atomic_write(path, content)
        # 2) Hot-swap in-memory state
        app_main.VENUE_PROFILES = profiles
        app_main.NAME_INDEX = build_name_index(profiles)

    return {"ok": True, "replaced": True, "path": path, "count": len(profiles)}
