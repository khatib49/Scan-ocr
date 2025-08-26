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

async def ping_mongo_or_raise():
    try:
        await mongo_client.admin.command("ping")
    except Exception as e:
        raise RuntimeError(f"Could not connect to MongoDB. Check MONGO_URL/MONGO_DB. Details: {e}")

# app/utils/logger.py (only the log_scan_invoice function shown)
async def log_scan_invoice(
    b64_image: Optional[str],
    merchant_guess: Optional[str],
    address_guess: Optional[str],
    profile: Optional[Dict[str, Any]],
    raw_text: Optional[str],
    parsed_data: Optional[Dict[str, Any]],
    final_result: Optional[Dict[str, Any]]
):
    try:
        doc = {
            "created_at": datetime.utcnow(),
            "image_url": b64_image,        # SAS URL expected
            "merchant_guess": merchant_guess,
            "address_guess": address_guess,
            "matched_profile": profile,
            "openai_raw": raw_text,
            "parsed_data": parsed_data,
            "final_result": final_result
        }
        res = await scan_invoice_collection.insert_one(doc)
        print(f"[mongo] inserted log _id={res.inserted_id} image_url={(b64_image or 'None')[:120]}")
    except Exception as e:
        print("[log_scan_invoice ERROR]", str(e))


async def log_error(
    b64_image: Optional[str],
    error: str,
    stage: str,
    extra: Optional[Dict[str, Any]] = None
):
    try:
        await error_collection.insert_one({
            "created_at": datetime.utcnow(),
            "image_url": b64_image,
            "stage": stage,
            "error": error,
            "extra": extra or {}
        })
    except Exception as e:
        print("[log_error ERROR]", str(e))
