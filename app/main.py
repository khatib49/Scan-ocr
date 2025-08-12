import os, json, base64
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from .venue_matcher import load_profiles, find_best_profile
from .prompts import SYSTEM_PROMPT
from utils.qr import decode_zatca_qr
from utils.transforms import coerce_number, coerce_nullish, norm_date, validate_and_score

# Load environment variables
try:
    load_dotenv()
except Exception:
    pass

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
    except:
        merchant_guess, addr_guess = "", ""

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

    # Post-process and patch with ZATCA QR
    d = data.get("data", {})
    for k in ("MerchantName","MerchantAddress","TransactionDate","StoreID","InvoiceId","CR","TaxID"):
        d[k] = coerce_nullish(d.get(k))
    d["TransactionDate"] = norm_date(d.get("TransactionDate"))

    qr_fields = decode_zatca_qr(raw)
    if qr_fields:
        if qr_fields.get("vat"):
            d["TaxID"] = qr_fields["vat"]
        if qr_fields.get("timestamp"):
            d["TransactionDate"] = qr_fields["timestamp"]
        if qr_fields.get("total") is not None and coerce_number(d.get("Total")) is None:
            d["Total"] = qr_fields["total"]
        if qr_fields.get("vat_amount") is not None and coerce_number(d.get("Tax")) is None:
            d["Tax"] = qr_fields["vat_amount"]

    data["data"] = d
    final_payload = validate_and_score(data, profile)
    return AnalyzeResponse(**final_payload)
