import os, json, base64
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from .venue_matcher import load_profiles, find_best_profile
from utils.qr import decode_zatca_qr
from utils.transforms import coerce_number, coerce_nullish, norm_date, validate_and_score
from utils.logger import log_scan_invoice, log_error

# Load environment variables
try:
    load_dotenv()
except Exception:
    pass

PROMPT_PATH = os.getenv("PROMPT_PATH", "data/prompt.txt")
with open(PROMPT_PATH, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment or .env")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Scan Invoice API", version="0.1.1")

# Load venue profiles from JSON
VENUE_PROFILES = load_profiles(os.getenv("VENUE_PROFILES_PATH", "data/venue_profiles.json"))

class AnalyzeResponse(BaseModel):
    data: Dict[str, Any]

def build_system_prompt(with_profile: Optional[Dict[str, Any]]) -> str:
    base = SYSTEM_PROMPT.strip()
    if with_profile:
        hints = with_profile.get("ExtractionHints") or {}
        slim = {
            "ExtractionHints": {
                k: v for k, v in hints.items()
                if k in {
                    "Language","Total_Label","Subtotal_Label","Tax_Label","CR_Label","TaxID_Label",
                    "Date_Label","Time_Label","Date_Format","Time_Format",
                    "InvoiceId_Label","StoreID_Label",
                    "MerchantName_Keyword","MerchantAddress_Keyword"
                } and v
            },
            "MerchantName_Keyword": with_profile.get("MerchantName_Keyword"),
            "MerchantId": with_profile.get("MerchantId"),
            "MerchantAddress_Keyword": with_profile.get("MerchantAddress_Keyword"),
            "SpendingRange": with_profile.get("Spending Range (SAR)")
        }
        base += "\n\n---\nCONTEXT VENUE PROFILE (for hints only; do not overwrite image values):\n" + json.dumps(slim, ensure_ascii=False)
    return base

@app.get("/health")
def health():
    return {"status": "ok", "profiles": len(VENUE_PROFILES)}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        if not raw:
            raise HTTPException(400, "Empty file.")
        b64 = base64.b64encode(raw).decode("utf-8")
    finally:
        await image.close()

    # Quick model call to guess merchant/address
    quick_prompt = """
Return ONLY this JSON:
{"m": "merchant name or null", "a": "merchant address or null"}
"""
    quick = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role":"system","content":"Read the image and return merchant + address only as JSON. DO NOT add text."},
            {"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]},
            {"role":"user","content":quick_prompt}
        ]
    )
    try:
        ma = json.loads(quick.choices[0].message.content or "{}")
        merchant_guess = (ma.get("m") or "")[:200]
        addr_guess = (ma.get("a") or "")[:200]
    except Exception as e:
        merchant_guess, addr_guess = "", ""
        await log_error(b64, str(e), "quick_guess")

    profile = find_best_profile(VENUE_PROFILES, merchant_guess, addr_guess)
    sys = build_system_prompt(profile)

    # Main extraction
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        messages=[
            {"role":"system","content": sys},
            {"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}
        ]
    )

    raw_txt = resp.choices[0].message.content or ""
    try:
        data = json.loads(raw_txt)
        if "data" not in data:
            raise ValueError("Missing 'data' root.")
    except Exception as e:
        await log_error(b64, str(e), "parse_openai_response", {"raw_response": raw_txt})
        data = {
            "data": {
                "MerchantName": None,
                "MerchantAddress": None,
                "TransactionDate": None,
                "StoreID": None,
                "InvoiceId": None,
                "CR": None,
                "TaxID": None,
                "Subtotal": None,
                "Tax": None,
                "Total": None,
                "fraudScore": 0,
                "confidentScore": 0,
                "reason": f"Model returned non-JSON or invalid format. {str(e)}"
            }
        }

    # d = data.get("data", {})
    # for k in ("MerchantName","MerchantAddress","TransactionDate","StoreID","InvoiceId","CR","TaxID"):
    #     d[k] = coerce_nullish(d.get(k))
    # d["TransactionDate"] = norm_date(d.get("TransactionDate"))

    # data["data"] = d
    final_payload = validate_and_score(data, profile)

    # Log final result
    await log_scan_invoice(
        b64_image=b64,
        merchant_guess=merchant_guess,
        address_guess=addr_guess,
        profile=profile,
        raw_text=raw_txt,
        parsed_data=data,
        final_result=final_payload
    )

    return AnalyzeResponse(**final_payload)
