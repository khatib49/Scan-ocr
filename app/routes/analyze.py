# app/routes/analyze.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, Dict, Any
import json

from ..prompt_store import get_active_prompt            # reads app/data/system_prompt.txt
from ..utils.images import to_base64_optimized
from ..utils.cost_controls import merge_usage
from ..db import get_db
from ..services.validators import validate_extracted
from ..services.openai_client import get_client         # use the same OpenAI client instance
from ..services.venue_matcher import match_venue_profile, load_profiles

router = APIRouter(prefix="/analyze", tags=["analyze"])

def build_system_prompt(base_prompt: str, profile: Optional[Dict[str, Any]]) -> str:
    base = (base_prompt or "").strip()
    if profile:
        hints = profile.get("ExtractionHints") or {}
        slim = {
            "ExtractionHints": {
                k: v for k, v in hints.items()
                if v and k in {
                    "Language","Total_Label","Subtotal_Label","Tax_Label","CR_Label","TaxID_Label",
                    "Date_Label","Time_Label","Date_Format","Time_Format","InvoiceId_Label","StoreID_Label",
                    "MerchantName_Keyword","MerchantAddress_Keyword"
                }
            },
            "MerchantName_Keyword": profile.get("MerchantName_Keyword"),
            "MerchantAddress_Keyword": profile.get("MerchantAddress_Keyword"),
            "SpendingRange": profile.get("Spending Range (SAR)")
        }
        base += "\n\n---\nCONTEXT VENUE PROFILE (for hints only; do not overwrite image values):\n" + json.dumps(slim, ensure_ascii=False)
    return base


@router.post("/image")
async def analyze_image(file: UploadFile = File(...), reference: Optional[str] = None):
    """
    TWO-CALL PIPELINE:
      Call #1: Probe merchant/address -> match profile (RapidFuzz inside match_venue_profile)
      Call #2: Main extraction with base prompt + appended profile hints
      Then: code-side validation, log to Mongo, return response
    """
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file.")
        image_b64 = to_base64_optimized(raw)

        # ===== Call #1: Probe merchant/address =====
        client = get_client()
        probe_prompt = (
            "Return ONLY this JSON with two keys and nothing else:\n"
            '{"m": "merchant name or null", "a": "merchant address or null"}'
        )

        probe_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Read the receipt image and extract ONLY merchant and address as JSON. No prose."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": probe_prompt},
                        {"type": "input_image", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}" }},
                    ],
                },
            ],
        )
        try:
            pa = json.loads(probe_resp.choices[0].message.content or "{}")
            merchant_guess = (pa.get("m") or "")[:200]
            addr_guess     = (pa.get("a") or "")[:200]
        except Exception:
            merchant_guess, addr_guess = "", ""

        # RapidFuzz venue/profile match (your adapter handles loading profiles)
        profile = match_venue_profile({"MerchantName": merchant_guess, "MerchantAddress": addr_guess})

        # ===== Call #2: Main extraction with profile hints =====
        base_prompt = await get_active_prompt()              # file-backed prompt text
        system_prompt = build_system_prompt(base_prompt, profile)

        extract_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the required fields. Return ONLY valid JSON under a single top-level object."},
                        {"type": "input_image", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}" }},
                    ],
                },
            ],
        )
        extract_text = extract_resp.choices[0].message.content or "{}"
        try:
            extracted_json = json.loads(extract_text)
        except Exception:
            extracted_json = {"data": {}}

        data_json: Dict[str, Any] = extracted_json.get("data") or extracted_json

        # ===== Code-side validation (no extra tokens) =====
        validation = validate_extracted(data_json)

        # Token usage aggregation (probe + extract)
        usage = {}
        try:
            u1 = probe_resp.usage.model_dump() if hasattr(probe_resp, "usage") else {}
            u2 = extract_resp.usage.model_dump() if hasattr(extract_resp, "usage") else {}
            usage = merge_usage(u1, u2)
        except Exception:
            pass

        # ===== Persist to Mongo =====
        db = get_db()
        await db["analyses"].insert_one({
            "reference": reference,
            "filename": file.filename,
            "probe": {"merchant": merchant_guess, "address": addr_guess},
            "profile_matched": bool(profile and profile.get("MerchantName_Keyword")),
            "extracted_raw": extract_text,
            "extracted_parsed": data_json,
            "validation": validation,
            "usage": usage,
        })

        # ===== Response =====
        return {
            "data": data_json,
            "validation": validation,
            "reason": validation.get("issues", []) and "See issues list." or "Checks passed.",
            "usage": usage,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
