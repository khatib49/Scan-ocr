# app/security.py
import os
from typing import List, Set, Optional, Dict, Any
from fastapi import HTTPException, Security, FastAPI, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load .env if present
try:
    load_dotenv()
except Exception:
    pass

API_KEY_NAME = "X-API-Key"

_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# --------- Mongo for Project lookup ----------
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB = os.getenv("MONGO_DB", "scan-invoice")

_mongo_client: Optional[AsyncIOMotorClient] = AsyncIOMotorClient(MONGO_URL) if MONGO_URL else None
_mongo_db = _mongo_client[MONGO_DB] if _mongo_client else None

# tiny in-memory cache to avoid a query every call
_PROJECT_CACHE: Dict[str, Dict[str, Any]] = {}

async def verify_api_key(request: Request, api_key: str = Security(_api_key_header)) -> str:
    """
    Dependency to enforce API-key auth.
    1) Header must be present
    2) (Optional) If API_KEYS is set, api_key must be in it
    3) api_key must map to a Project document (Project.ApiKey) in Mongo
    On success, attaches `request.state.project` with {_id, Name}.
    """
    if not api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    # Env allow-list (kept for extra safety). If you want Mongo-only, delete this block.
    # if _API_KEYS and api_key not in _API_KEYS:
    #     raise HTTPException(status_code=403, detail="Invalid or missing API key")

    if _mongo_db is None:
        raise HTTPException(status_code=500, detail="MongoDB not configured")

    proj = _PROJECT_CACHE.get(api_key)
    if proj is None:
        proj_doc = await _mongo_db["Project"].find_one({"ApiKey": api_key}, {"_id": 1, "Name": 1})
        if not proj_doc:
            raise HTTPException(status_code=403, detail="API key not assigned to a project")
        proj = {"_id": proj_doc["_id"], "Name": proj_doc.get("Name")}
        _PROJECT_CACHE[api_key] = proj

    # make the project available to handlers without another query
    request.state.project = proj
    return api_key

def add_cors(app: FastAPI) -> None:
    """
    Attach CORS using CORS_ALLOWED_ORIGINS env.
    - CORS_ALLOWED_ORIGINS="*" -> allow all origins (credentials disabled)
    - CORS_ALLOWED_ORIGINS="http://localhost:3000,https://myapp.com" -> allow list (credentials enabled)
    """
    origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "*").strip()
    if origins_env == "*":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[API_KEY_NAME],
        )
    else:
        origins = [o.strip() for o in origins_env.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[API_KEY_NAME],
        )
