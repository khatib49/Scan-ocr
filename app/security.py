# app/security.py
import os
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, FastAPI, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

try:
    load_dotenv()
except Exception:
    pass

# ---- Public (project) API key ----
API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(
    name=API_KEY_NAME,
    scheme_name="ProjectApiKey",   
    auto_error=False,
)

# ---- Admin API key (NEW) ----
# Admin key
ADMIN_KEY_NAME = "X-Admin-Key"
_admin_key_header = APIKeyHeader(
    name=ADMIN_KEY_NAME,
    scheme_name="AdminApiKey",
    auto_error=False,
)
# --------- Mongo for Project lookup ----------
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB = os.getenv("MONGO_DB", "scan-invoice")

_mongo_client: Optional[AsyncIOMotorClient] = AsyncIOMotorClient(MONGO_URL) if MONGO_URL else None
_mongo_db = _mongo_client[MONGO_DB] if _mongo_client else None

_PROJECT_CACHE: Dict[str, Dict[str, Any]] = {}

async def verify_api_key(request: Request, api_key: str = Security(_api_key_header)) -> str:
    if not api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    if _mongo_db is None:
        raise HTTPException(status_code=500, detail="MongoDB not configured")

    proj = _PROJECT_CACHE.get(api_key)
    if proj is None:
        proj_doc = await _mongo_db["Project"].find_one({"ApiKey": api_key}, {"_id": 1, "Name": 1})
        if not proj_doc:
            raise HTTPException(status_code=403, detail="API key not assigned to a project")
        proj = {"_id": proj_doc["_id"], "Name": proj_doc.get("Name")}
        _PROJECT_CACHE[api_key] = proj

    request.state.project = proj
    return api_key

# ---- Admin key validator----
async def verify_admin_key(request: Request, admin_key: str = Security(_admin_key_header)) -> str:
    if not admin_key:
        raise HTTPException(status_code=403, detail="Invalid or missing Admin API key")
    if _mongo_db is None:
        raise HTTPException(status_code=500, detail="MongoDB not configured")

    cache_key = f"admin:{admin_key}"            # avoid collisions with project keys
    proj = _PROJECT_CACHE.get(cache_key)
    if proj is None:
        # if your admin doc is literally named "admin," include it in $in; otherwise keep "admin".
        proj_doc = await _mongo_db["Project"].find_one(
            {"ApiKey": admin_key, "Name": {"$in": ["admin", "admin,"]}},
            {"_id": 1, "Name": 1},
        )
        if not proj_doc:
            raise HTTPException(status_code=403, detail="API key not assigned to a project (admin)")
        proj = {"_id": proj_doc["_id"], "Name": proj_doc.get("Name")}
        _PROJECT_CACHE[cache_key] = proj

    request.state.project = proj
    return admin_key

def add_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # allow all origins
        allow_credentials=True,
        allow_methods=["*"],  # allow all methods (GET, POST, PUT, DELETE, etc.)
        allow_headers=["*"],  # allow all headers
    )
