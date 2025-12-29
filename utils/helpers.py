# app/helpers/projects.py
import os, secrets
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from bson import ObjectId
from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, errors

# ---------- Lookups ----------

async def get_project_by_api_key(db: AsyncIOMotorDatabase, api_key: str) -> Dict[str, Any]:
    project = await db["Project"].find_one({"ApiKey": api_key})
    if not project:
        raise HTTPException(status_code=403, detail="API key not assigned to a project")
    return project

# ---------- Indexes & utilities ----------

from pymongo import ASCENDING, errors

async def ensure_project_indexes(db) -> None:
    """
    Idempotent index bootstrap for critical collections.

    - Project:
        * Ensures unique index on ApiKey
    - VenueProfile:
        * Ensures compound text index for fuzzy merchant matching
    """
    # -------------------------------
    # 1) Ensure Project.ApiKey unique
    # -------------------------------
    project_coll = db["Project"]

    try:
        info = await project_coll.index_information()
    except Exception:
        info = {}

    existing_name = None
    existing_unique = False

    for name, spec in (info or {}).items():
        key = spec.get("key")
        if key == [("ApiKey", 1)] or key == (("ApiKey", 1),):
            existing_name = name
            existing_unique = bool(spec.get("unique", False))
            break

    if existing_name:
        if not existing_unique:
            try:
                await project_coll.drop_index(existing_name)
            except errors.OperationFailure:
                pass  # ignore race
            await project_coll.create_index([("ApiKey", ASCENDING)], unique=True)
    else:
        try:
            await project_coll.create_index([("ApiKey", ASCENDING)], unique=True)
        except errors.OperationFailure as e:
            if "IndexOptionsConflict" not in str(e):
                raise

    # ---------------------------------
    # 2) Ensure VenueProfile text index
    # ---------------------------------
    venue_coll = db["VenueProfile"]
    try:
        vinfo = await venue_coll.index_information()
    except Exception:
        vinfo = {}

    # Check if a text index already exists (any name)
    existing_text = next(
        (n for n, spec in vinfo.items() if any("text" in str(k[1]) for k in spec.get("key", []))),
        None,
    )

    if not existing_text:
        try:
            await venue_coll.create_index(
                [
                    ("MerchantName_Keyword", "text"),
                    ("TenantName", "text"),
                    ("Brand", "text"),
                    ("Aliases", "text"),
                ],
                name="venue_text_idx",
                default_language="none",
                language_override="none",
                weights={
                    "MerchantName_Keyword": 10,
                    "Brand": 8,
                    "TenantName": 5,
                    "Aliases": 5,
                },
            )
            print("[index] Created VenueProfile text index (venue_text_idx).")
        except errors.OperationFailure as e:
            if "IndexOptionsConflict" not in str(e):
                raise
    else:
        print(f"[index] VenueProfile text index already exists: {existing_text}")


async def generate_unique_api_key(db: AsyncIOMotorDatabase) -> str:
    """Generate a collision-resistant API key and ensure uniqueness."""
    for _ in range(10):
        key = secrets.token_urlsafe(48)  # ~64 chars, safe for headers
        if not await db["Project"].find_one({"ApiKey": key}, {"_id": 1}):
            return key
    raise RuntimeError("Could not generate unique API key after several attempts")

def normalize_project_out(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Mongo document to clean API shape."""
    return {
        "_id": str(doc["_id"]),
        "Name": doc.get("Name"),
        "ApiKey": doc.get("ApiKey"),
        "CreatedAt": doc.get("CreatedAt"),
    }

def parse_oid(project_id: str) -> ObjectId:
    try:
        return ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project id")


#uvicorn app.main:app --host 0.0.0.0 --port 8000
#uvicorn app.main_gemini:app --host 0.0.0.0 --port 8000