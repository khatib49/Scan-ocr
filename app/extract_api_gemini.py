# app/extract_api_gemini.py
import os, json, base64, re
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Security
from fastapi.responses import JSONResponse
from app.main_gemini import call_gemini_with_image
from app.security import verify_admin_key

router = APIRouter(prefix="/extract", tags=["extract"])


def _data_dir() -> str:
    """Return the data directory path"""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def _read_prompt_template() -> Optional[str]:
    """Read the extract prompt template from data/extract_prompt.txt"""
    p = os.path.join(_data_dir(), "extract_prompt.txt")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None

def _inject_merchant_id(prompt: str, merchant_id: str) -> str:
    """Replace {{MERCHANT_ID}} placeholder with actual merchant ID"""
    return prompt.replace("{{MERCHANT_ID}}", merchant_id)


@router.post(
    "",
    dependencies=[Security(verify_admin_key)],
    summary="Extract receipt using Gemini"
)
async def extract_with_gemini(
    merchantId: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    if not image:
        raise HTTPException(400, "Image is required")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Empty image")

    # Read prompt template or use fallback
    prompt_text = _read_prompt_template() or """
You are a precise data extractor for receipts. Read the image and output ONE JSON object
in the EXACT schema below. Do not add wrapper keys (like "data" or "result"). Do not add
extra fields. If a field is unknown, use "" (or null for the fields marked null).

Return ONLY the JSON object, nothing else.

Schema:
{
  "MerchantName_Keyword": ["<primary name>", "<alt English>", "<alt Arabic>"],
  "MerchantAddress_Keyword": ["<address variants, e.g. city/area/branch>"],
  "MerchantId": {{MERCHANT_ID}},
  "Category": "<F&B|Retail|Entertainment|Services>",
  "Sub-category": "<e.g., Coffee Shop | Restaurant | Perfumes>",
  "Spending Range (SAR)": "<e.g., 10-100>",
  "ExtractionHints": {
    "Language": "Mixed",
    "Total_Label": "<as printed on receipt>",
    "Subtotal_Label": "<as printed on receipt>",
    "Tax_Label": "<as printed on receipt>",
    "CR_Label": null,
    "TaxID_Label": "<VAT or tax reg>",
    "Date_Label": "<date as printed>",
    "Time_Label": "<time as printed>",
    "Date_Format": "YYYY/MM/DD",
    "Time_Format": "HH:mm",
    "InvoiceId_Label": "<invoice/check/order id>",
    "StoreID_Label": null,
    "MerchantName_Keyword": ["<repeat same list as root>"],
    "MerchantAddress_Keyword": ["<repeat same list as root>"]
  }
}

Guidelines:
- Use EXACT label strings from the receipt for *_Label fields if visible; otherwise "".
- Include both Arabic and English variants seen on the image for merchant & address arrays.
- Do NOT invent values (leave ""/null if not visible).
- MerchantId MUST equal the injected value.
"""
    
    # Inject merchant ID
    prompt_text = _inject_merchant_id(prompt_text, merchantId.strip())

    try:
        response, _ = await call_gemini_with_image(
            prompt=prompt_text,
            image_bytes=raw,
            mime_type=image.content_type or "image/jpeg",
            temp=0.1,
            call_type="extract"
        )
    except Exception as e:
        raise HTTPException(502, f"Gemini error: {e}")

    try:
        data = json.loads(response)
    except Exception:
        raise HTTPException(502, "Gemini returned invalid JSON")

    return JSONResponse(content=data)
