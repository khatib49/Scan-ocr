# app/routes/analyze.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, Dict, Any
import json

from ..schemas import AnalyzeResponse
from ..prompt_store import get_active_prompt
from ..services.openai_client import gpt_extract, gpt_reason
from ..services.validators import validate_extracted
from ..utils.images import to_base64_optimized
from ..utils.cost_controls import merge_usage
from ..db import get_db

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...), reference: Optional[str] = None):
    """
    Orchestrates the end-to-end analysis:
      1) Read and optimize image -> base64
      2) Fetch dynamic system prompt (from file via prompt_store)
      3) GPT extract (multimodal) -> JSON (fallback if invalid)
      4) Code-side validation (math/VAT/venue)
      5) GPT reason (text-only, cheap) -> short explanation JSON
      6) Persist both calls + validation + usage into Mongo
      7) Return merged response
    """
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file.")
        image_b64 = to_base64_optimized(raw)

        # 1) Load the current system prompt from file
        system_prompt = await get_active_prompt()

        # 2) GPT extraction (image)
        extract_res = await gpt_extract(image_b64, system_prompt)
        extract_text = extract_res.get("text", "{}")

        try:
            extracted_json = json.loads(extract_text or "{}")
        except Exception:
            # fallback minimal structure if model returns non-JSON
            extracted_json = {"data": {}}

        # Allow both shapes: { "data": {...} } or flat {...}
        data_json: Dict[str, Any] = extracted_json.get("data") or extracted_json

        # 3) Code-side validation
        validation = validate_extracted(data_json)

        # 4) GPT reason (text-only, cheaper)
        reason_res = await gpt_reason(data_json, validation, system_prompt)
        reason_text = reason_res.get("text", "{}")
        try:
            reason_json = json.loads(reason_text or "{}")
        except Exception:
            reason_json = {"reason": "Validation completed."}

        # 5) Aggregate usage tokens from both calls
        usage = merge_usage(extract_res.get("usage", {}), reason_res.get("usage", {}))

        # 6) Persist both calls & final into Mongo
        db = get_db()
        doc = {
            "reference": reference,
            "filename": file.filename,
            "extracted_raw": extract_text,
            "extracted_parsed": data_json,
            "validation": validation,
            "reason": reason_json.get("reason"),
            "usage": usage,
        }
        await db["analyses"].insert_one(doc)

        # 7) Return final response
        return {
            "data": data_json,
            "validation": validation,
            "reason": reason_json.get("reason"),
            "usage": usage,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Surface as 500 with message
        raise HTTPException(status_code=500, detail=str(e))
