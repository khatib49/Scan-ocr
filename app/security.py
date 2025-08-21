# app/security.py
import os
from typing import List, Set
from fastapi import HTTPException, Security, FastAPI
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env if present
try:
    load_dotenv()
except Exception:
    pass

API_KEY_NAME = "X-API-Key"

def _parse_env_csv(name: str) -> List[str]:
    raw = os.getenv(name, "") or ""
    return [x.strip() for x in raw.split(",") if x.strip()]

def _load_api_keys() -> Set[str]:
    keys = set(_parse_env_csv("API_KEYS"))
    # Optional fallback single key (backward-compat)
    single = os.getenv("SCAN_API_KEY")
    if single:
        keys.add(single.strip())
    return keys

_API_KEYS: Set[str] = _load_api_keys()
_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    Dependency to enforce API-key auth.
    Accepts any key in API_KEYS (comma-separated) or SCAN_API_KEY.
    """
    if api_key and api_key in _API_KEYS:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid or missing API key")

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
            allow_credentials=False,  # credentials not allowed with wildcard per browsers
            allow_methods=["*"],
            allow_headers=["*"],      # or ["Content-Type", "Accept", API_KEY_NAME]
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
