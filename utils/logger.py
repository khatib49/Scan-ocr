import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables (this is CRITICAL)
load_dotenv()

# Mongo connection
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise RuntimeError("MONGO_URL not found in environment.")

mongo_client = AsyncIOMotorClient(MONGO_URL)
mongo_db = mongo_client["scan-invoice"]

# Collections
scan_invoice_collection = mongo_db["invoice"]  # successful logs
error_collection = mongo_db["logs"]            # failure logs

# Log successful analysis
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
        await scan_invoice_collection.insert_one({
            "created_at": datetime.utcnow(),
            "image": b64_image,
            "merchant_guess": merchant_guess,
            "address_guess": address_guess,
            "matched_profile": profile,
            "openai_raw": raw_text,
            "parsed_data": parsed_data,
            "final_result": final_result
        })
    except Exception as e:
        print("[log_scan_invoice ERROR]", str(e))

# Log failed/incomplete stages
async def log_error(
    b64_image: Optional[str],
    error: str,
    stage: str,
    extra: Optional[Dict[str, Any]] = None
):
    try:
        await error_collection.insert_one({
            "created_at": datetime.utcnow(),
            "image": b64_image,
            "stage": stage,
            "error": error,
            "extra": extra or {}
        })
    except Exception as e:
        print("[log_error ERROR]", str(e))
