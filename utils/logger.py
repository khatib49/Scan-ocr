import os
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

try:
    load_dotenv()
except Exception:
    pass

MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB = os.getenv("MONGO_DB", "scan-invoice")

if not MONGO_URL:
    raise RuntimeError("MONGO_URL not found in environment.")

mongo_client = AsyncIOMotorClient(MONGO_URL)
mongo_db = mongo_client[MONGO_DB]

# Collections
scan_invoice_collection = mongo_db["invoice"]  # successful logs
error_collection = mongo_db["logs"]            # failure logs
telemetry_collection    = mongo_db["telemetry_requests"] # telemetry and events

async def ping_mongo_or_raise():
    try:
        await mongo_client.admin.command("ping")
    except Exception as e:
        raise RuntimeError(f"Could not connect to MongoDB. Check MONGO_URL/MONGO_DB. Details: {e}")

# app/utils/logger.py (only the log_scan_invoice function shown)
async def log_scan_invoice(
    imageUrl: Optional[str],
    merchant_guess: Optional[str],
    address_guess: Optional[str],
    profile: Optional[Dict[str, Any]],
    raw_text: Optional[str],
    userReference: str,
    final_result: Optional[Dict[str, Any]],
    request_id: Optional[str] = None,
    scanReference: Optional[str] = None
):
    try:
        doc = {
            "created_at": datetime.utcnow(),
            "image_url": imageUrl,       
            "merchant_guess": merchant_guess,
            "address_guess": address_guess,
            "matched_profile": profile,
            "openai_raw": raw_text,
            "userReference" :  userReference,
            "final_result": final_result,
            "request_id": request_id,
            "scanReference": scanReference
        }
        res = await scan_invoice_collection.insert_one(doc)
    except Exception as e:
        print("[log_scan_invoice ERROR]", str(e))


async def log_error(
    imageUrl: Optional[str],
    error: str,
    stage: str,
    userReference : str,
    extra: Optional[Dict[str, Any]] = None,
    scanReference: Optional[str] = None
):
    try:
        await error_collection.insert_one({
            "created_at": datetime.utcnow(),
            "image_url": imageUrl,
            "stage": stage,
            "error": error,
            "userReference" : userReference,
            "extra": extra or {}
        })
    except Exception as e:
        print("[log_error ERROR]", str(e))




# ---------- ONE-DOC-PER-REQUEST API ----------
async def ensure_telemetry_indexes():
    try:
        await telemetry_collection.create_index([("request_id", 1)], unique=True)
        await telemetry_collection.create_index([("created_at", -1)])
        await telemetry_collection.create_index([("userReference", 1), ("created_at", -1)])
        await telemetry_collection.create_index([("status", 1), ("created_at", -1)])
    except Exception as e:
        print("[ensure_telemetry_indexes ERROR]", str(e))

async def init_request_log(request_id: str, path: str, userReference: Optional[str], 
    scanReference: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    try:
        now = datetime.utcnow()
        await telemetry_collection.update_one(
            {"request_id": request_id},
            {
                "$setOnInsert": {
                    "request_id": request_id,
                    "kind": "analyze_request",
                    "created_at": now,
                    "path": path,
                    "userReference": userReference,
                    "scanReference": scanReference,
                    "meta": meta or {},
                    "llm_calls": [],
                    "blob_ops": [],     # <-- NEW
                    "status": "started",
                },
                "$set": {"started_at": now},
            },
            upsert=True,
        )
    except Exception as e:
        print("[init_request_log ERROR]", str(e))

async def append_blob_op(
    request_id: str,
    op: str,                 # "upload_image_bytes" | "build_read_url" | etc.
    duration_ms: float,
    success: bool,
    meta: Optional[Dict[str, Any]] = None
):
    """Push a blob-storage operation timing into the request doc."""
    try:
        await telemetry_collection.update_one(
            {"request_id": request_id},
            {
                "$push": {
                    "blob_ops": {
                        "at": datetime.utcnow(),
                        "op": op,
                        "duration_ms": duration_ms,
                        "success": success,
                        "meta": meta or {}
                    }
                }
            },
            upsert=True,
        )
    except Exception as e:
        print("[append_blob_op ERROR]", str(e))


async def init_request_log(
    request_id: str,
    path: str,
    userReference: Optional[str],
    scanReference: Optional[str] = None,  
    meta: Optional[Dict[str, Any]] = None,
):
    """Create (or upsert) the request log with start info."""
    try:
        now = datetime.utcnow()
        await telemetry_collection.update_one(
            {"request_id": request_id},
            {
                # set once on first insert
                "$setOnInsert": {
                    "request_id": request_id,
                    "kind": "analyze_request",
                    "created_at": now,
                    "path": path,
                    "userReference": userReference,
                    "scanReference": scanReference,
                    "meta": meta or {},
                    "llm_calls": [],
                    "status": "started",
                },
                # always update these on every init (in case of retries)
                "$set": {
                    "started_at": now,
                },
            },
            upsert=True,
        )
    except Exception as e:
        print("[init_request_log ERROR]", str(e))

async def append_llm_call(
    request_id: str,
    call_type: str,           # "quick" | "main" | etc.
    model: Optional[str],
    duration_ms: float,
    usage: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    """Push a per-LLM-call record into the request doc."""
    try:
        await telemetry_collection.update_one(
            {"request_id": request_id},
            {
                "$push": {
                    "llm_calls": {
                        "at": datetime.utcnow(),
                        "call_type": call_type,
                        "model": model,
                        "duration_ms": duration_ms,
                        "usage": usage or {},
                        "extra": extra or {},
                    }
                }
            },
            upsert=True,  # if somehow missing, create the shell doc
        )
    except Exception as e:
        print("[append_llm_call ERROR]", str(e))

async def finalize_request_log(
    request_id: str,
    success: bool,
    total_ms: float,
    summary: Optional[Dict[str, Any]] = None,
):
    """Mark request as finished, with totals."""
    try:
        now = datetime.utcnow()
        await telemetry_collection.update_one(
            {"request_id": request_id},
            {
                "$set": {
                    "ended_at": now,
                    "total_ms": total_ms,
                    "status": "success" if success else "error",
                    "summary": summary or {},
                }
            },
            upsert=True,
        )
    except Exception as e:
        print("[finalize_request_log ERROR]", str(e))

# ---------- existing logs (unchanged apart from optional userReference) ----------