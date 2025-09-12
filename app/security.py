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
    scheme_name="AdminApiKey",     # 👈 distinct name for Swagger
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

# ---- Admin key validator (NEW) ----
def verify_admin_key(admin_key: str = Security(_admin_key_header)) -> str:
    expected = (os.getenv("ADMIN_API_KEY") or "").strip()
    if not expected:
        # Misconfiguration is a server issue, not a client issue
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")
    if admin_key != expected:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return admin_key

def add_cors(app: FastAPI) -> None:
    origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "*").strip()
    expose = [API_KEY_NAME, ADMIN_KEY_NAME]  # (optional) expose both in responses
    if origins_env == "*":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],  # already allows X-Admin-Key in requests
            expose_headers=expose,
        )
    else:
        origins = [o.strip() for o in origins_env.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=expose,
        )
