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
    imageUrl: Optional[str],
    merchant_guess: Optional[str],
    address_guess: Optional[str],
    profile: Optional[Dict[str, Any]],
    raw_text: Optional[str],
    userReference: str,
    final_result: Optional[Dict[str, Any]]
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
            "final_result": final_result
        }
        res = await scan_invoice_collection.insert_one(doc)
    except Exception as e:
        print("[log_scan_invoice ERROR]", str(e))


async def log_error(
    imageUrl: Optional[str],
    error: str,
    stage: str,
    userReference : str,
    extra: Optional[Dict[str, Any]] = None
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
