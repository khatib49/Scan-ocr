# app/venue_profiles_api.py
import os, json, hashlib, asyncio, tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Body, Query, Security
from app.security import verify_admin_key
from app.venue_matcher import build_name_index

router = APIRouter(prefix="/venue-profiles", tags=["Venue Profiles"])

# Guard concurrent read/modify/write
_profiles_lock = asyncio.Lock()

# ---------- helpers ----------

def _profiles_path() -> str:
    return os.getenv("VENUE_PROFILES_PATH", "data/venue_profiles.json")

def _sha256_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _atomic_write(path: str, content: bytes) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as tf:
        tf.write(content)
        tmp = tf.name
    os.replace(tmp, path)

def _file_meta(path: str) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        return {"path": path, "exists": True, "size_bytes": st.st_size,
                "last_modified": datetime.fromtimestamp(st.st_mtime).isoformat()}
    except FileNotFoundError:
        return {"path": path, "exists": False}

def _load_from_disk() -> List[Dict[str, Any]]:
    path = _profiles_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise HTTPException(500, detail="Profiles file is not a JSON array")
    return data

def _save_to_disk_and_refresh(profiles: List[Dict[str, Any]]) -> None:
    # Validate index builds before writing
    build_name_index(profiles)
    content = json.dumps(profiles, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write(_profiles_path(), content)
    # Hot-swap globals used by /analyze
    from app import main as app_main
    app_main.VENUE_PROFILES = profiles
    app_main.NAME_INDEX = build_name_index(profiles)

def _norm_id(mid: Any) -> str:
    # compare MerchantId as strings to avoid int/str mismatch
    return str(mid)

def _find_idx_by_id(profiles: List[Dict[str, Any]], merchant_id: Any) -> int:
    target = _norm_id(merchant_id)
    for i, p in enumerate(profiles):
        if _norm_id(p.get("MerchantId")) == target:
            return i
    return -1

def _validate_profile_fields(p: Dict[str, Any], *, is_add: bool) -> None:
    # Required on add; optional on patch (if present, type-checked)
    def _req(name, typ):
        if name not in p:
            raise HTTPException(400, detail=f"Missing required field: {name}")
        if not isinstance(p[name], typ):
            raise HTTPException(400, detail=f"{name} must be of type {typ.__name__}")

    def _opt(name, typ):
        if name in p and not isinstance(p[name], typ):
            raise HTTPException(400, detail=f"{name} must be of type {typ.__name__}")

    if is_add:
        _req("MerchantId", (int, str))
        _req("MerchantName_Keyword", list)
        _req("MerchantAddress_Keyword", list)
    else:
        _opt("MerchantId", (int, str))  # we will forbid changing it below
        _opt("MerchantName_Keyword", list)
        _opt("MerchantAddress_Keyword", list)

# ---------- endpoints ----------

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
    raw = open(path, "rb").read()
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
    path = _profiles_path()
    if not os.path.exists(path):
        raise HTTPException(404, detail=f"File not found: {path}")
    async with _profiles_lock:
        profiles = _load_from_disk()
        _save_to_disk_and_refresh(profiles)  # validates & rebuilds index
    return {"ok": True, "reloaded": True, "count": len(profiles)}

# ---- CRUD by MerchantId ----

@router.get(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Get a merchant profile (admin)",
)
async def get_merchant(merchant_id: str) -> Dict[str, Any]:
    profiles = _load_from_disk()  # read-only
    idx = _find_idx_by_id(profiles, merchant_id)
    if idx < 0:
        raise HTTPException(404, detail="Merchant not found")
    return {"profile": profiles[idx]}

@router.post(
    "",
    dependencies=[Security(verify_admin_key)],
    summary="Add a merchant profile (admin)",
)
async def add_merchant(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    _validate_profile_fields(payload, is_add=True)
    async with _profiles_lock:
        profiles = _load_from_disk()
        if _find_idx_by_id(profiles, payload["MerchantId"]) >= 0:
            raise HTTPException(status_code=409, detail="MerchantId already exists")
        profiles.append(payload)
        _save_to_disk_and_refresh(profiles)
    return {"ok": True, "created": True, "profile": payload, "count": len(profiles)}

@router.patch(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Update a merchant profile by MerchantId (admin)",
)
async def update_merchant(
    merchant_id: str,
    patch: Dict[str, Any] = Body(..., description="Fields to update"),
) -> Dict[str, Any]:
    _validate_profile_fields(patch, is_add=False)
    if "MerchantId" in patch and _norm_id(patch["MerchantId"]) != _norm_id(merchant_id):
        raise HTTPException(400, detail="Cannot change MerchantId; use the existing id in path")

    async with _profiles_lock:
        profiles = _load_from_disk()
        idx = _find_idx_by_id(profiles, merchant_id)
        if idx < 0:
            raise HTTPException(404, detail="Merchant not found")

        updated = {**profiles[idx], **patch}
        # quick per-profile validation for required lists if present
        _validate_profile_fields(
            {k: v for k, v in updated.items() if k in {"MerchantName_Keyword", "MerchantAddress_Keyword"}},
            is_add=False,
        )

        profiles[idx] = updated
        _save_to_disk_and_refresh(profiles)
    return {"ok": True, "updated": True, "profile": updated}

@router.delete(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Delete a merchant profile by MerchantId (admin)",
)
async def delete_merchant(merchant_id: str) -> Dict[str, Any]:
    async with _profiles_lock:
        profiles = _load_from_disk()
        idx = _find_idx_by_id(profiles, merchant_id)
        if idx < 0:
            raise HTTPException(404, detail="Merchant not found")
        removed = profiles.pop(idx)
        _save_to_disk_and_refresh(profiles)
    return {"ok": True, "deleted": True, "removed": removed, "count": len(profiles)}
