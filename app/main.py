import os, json, base64
from typing import Optional, Dict, Any

from fastapi import FastAPI, Query, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from .venue_matcher import load_profiles, build_name_index, find_best_profile_indexed
from utils.transforms import coerce_number, coerce_nullish, norm_date, validate_and_score  # your module
from utils.logger import log_scan_invoice, log_error, ping_mongo_or_raise

from .security import verify_api_key, add_cors
from .blob_service import upload_image_bytes, assert_blob_ready , build_read_url

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

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Scan Invoice API", version="0.2.0", dependencies=[Depends(verify_api_key)])

# CORS
add_cors(app)

# Load venue profiles
VENUE_PROFILES = load_profiles(os.getenv("VENUE_PROFILES_PATH", "data/venue_profiles.json"))
NAME_INDEX = build_name_index(VENUE_PROFILES)

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

@app.on_event("startup")
async def _startup_checks():
    # Fail fast on Mongo; warn on Azure
    await ping_mongo_or_raise()
    try:
        await assert_blob_ready()
    except Exception as e:
        print("[startup] Azure Blob not ready:", str(e))

@app.get("/health")
def health():
    return {"status": "ok", "profiles": len(VENUE_PROFILES)}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    userReference : str = Query(..., description="Your internal user ID or reference"),
    save_image: bool = Query(False, description="If true, saves the uploaded image to Azure Blob Storage")
):
    # 1) Read file
    try:
        raw = await image.read()
        if not raw:
            raise HTTPException(400, "Empty file.")
        content_type = image.content_type
    finally:
        await image.close()

    # 2) Optionally save to Azure and get SAS URL
    blob_url = None
    blob_name = None
    if save_image:
        try:
            preferred = None
            if image.filename and len(image.filename) < 150 and "." in image.filename:
                preferred = image.filename.replace("\\", "/").split("/")[-1]
            blob_name, blob_url = await upload_image_bytes(
                raw,
                content_type=content_type,
                preferred_name=preferred,
            return_sas=True,    
            )
            if not blob_url and blob_name:
                # ✅ belt-and-suspenders fallback (shouldn’t happen, but makes it bulletproof)
                blob_url = await build_read_url(blob_name)

                
            # Optional: print for debugging
            print("[blob] SAS:", blob_url)
            print("[blob] blob_name:", blob_name)
        except Exception as e:
            await log_error("", f"Blob upload failed: {str(e)}", "blob_upload")
    print("[log] about to insert image_url:", blob_url)
    # 3) Quick pass to guess merchant/address (fast + cheap)
    quick_prompt_path = os.getenv("QUICK_PROMPT_PATH", "data/quick_prompt.txt")
    try:
        with open(quick_prompt_path, encoding="utf-8") as f:
            QUICK_PROMPT = f.read()
    except Exception:
        QUICK_PROMPT = '{"instruction":"Return JSON { \\"m\\": \\"<merchant>\\", \\"a\\": \\"<address>\\" } only."}'

    b64 = base64.b64encode(raw).decode("utf-8")

    quick = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {"role":"system","content":"Read the image and return merchant + address only as JSON. DO NOT add text."},
            {"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]},
            {"role":"user","content":QUICK_PROMPT}
        ]
    )

    try:
        ma = json.loads(quick.choices[0].message.content or "{}")
        merchant_guess = (ma.get("m") or "").strip()[:200]
        addr_guess = (ma.get("a") or "").strip()[:200]
    except Exception as e:
        merchant_guess, addr_guess = "", ""
        await log_error(blob_url, str(e), "quick_guess")

    # 4) Venue match
    match = find_best_profile_indexed(NAME_INDEX, merchant_guess)
    matched = match.get("matched")
    profile = match.get("profile")

    raw_txt = None
    data = None

    # 5) If no match, return minimal with high fraud score
    if not merchant_guess or not matched:
        data = {
            "data": {
                "MerchantName": merchant_guess or None,
                "MerchantAddress": addr_guess or None,
                "MerchantId": None,
                "TransactionDate": None,
                "StoreID": None,
                "InvoiceId": None,
                "CR": None,
                "TaxID": None,
                "Subtotal": None,
                "Tax": None,
                "Total": None,
                "fraudScore": 100,
                "confidentScore": 0,
                "reason": ("Merchant name missing." if not merchant_guess else "No matching venue profile found.")
            }
        }
        final_payload = data
    else:
        # 6) Main extraction
        sys = build_system_prompt(profile)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
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
            await log_error(blob_url, str(e), "parse_openai_response", {"raw_response": raw_txt})
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

        # Validate/score via your custom logic
        final_payload = validate_and_score(data, profile)

    # 7) Persist log (SAS URL included if saved)
    await log_scan_invoice(
        imageUrl=blob_url,
        merchant_guess=merchant_guess if 'merchant_guess' in locals() else None,
        address_guess=addr_guess if 'addr_guess' in locals() else None,
        profile=profile if 'profile' in locals() else None,
        raw_text=raw_txt,
        userReference = userReference,
        final_result=final_payload
    )

    return AnalyzeResponse(**final_payload)
