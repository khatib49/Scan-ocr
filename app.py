import os, json, base64, re, unicodedata
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from venueMatcher import load_profiles, find_best_profile, normalize_text

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "data/system_prompt.txt")
VENUE_PROFILES_PATH = os.getenv("VENUE_PROFILES_PATH", "data/venue_profiles.json")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment or .env")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Scan Invoice API", version="0.2.1")

# Load profiles into memory
VENUE_PROFILES = load_profiles(VENUE_PROFILES_PATH)

# Response schema
class AnalyzeResponse(BaseModel):
    data: Dict[str, Any]

# Utility functions
NULL_STRINGS = {"", "null", "none", "nil", "n/a", "na", "—", "-", "غير متوفر", "غير موجود"}

def coerce_nullish(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    return None if s in NULL_STRINGS else x

def norm_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip().replace("  ", " ")
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    s = s.translate(trans)
    fmts = [
        "%d/%m/%Y %I:%M %p", "%d/%m/%Y %H:%M",
        "%Y/%m/%d %I:%M:%S %p", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
        "%d/%m/%Y", "%Y/%m/%d",
        "%d-%m-%Y %H:%M", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    try:
        s2 = s.replace("T", " ").replace("Z", "").split(".")[0]
        dt = datetime.fromisoformat(s2)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return s

def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise RuntimeError(f"System prompt file not found: {SYSTEM_PROMPT_PATH}")

def build_system_prompt(with_profile: Optional[Dict[str, Any]]) -> str:
    base = load_system_prompt().strip()
    if with_profile:
        hints = with_profile.get("ExtractionHints") or {}
        slim = {
            "ExtractionHints": {k: v for k, v in hints.items() if v},
            "MerchantName_Keyword": with_profile.get("MerchantName_Keyword"),
            "MerchantAddress_Keyword": with_profile.get("MerchantAddress_Keyword"),
            "SpendingRange": with_profile.get("Spending Range (SAR)")
        }
        base += "\n\n---\nCONTEXT VENUE PROFILE:\n" + json.dumps(slim, ensure_ascii=False)
    return base

def supports_temperature(model_name: str) -> bool:
    """GPT-5 models do not allow setting temperature, GPT-4o does."""
    return not model_name.lower().startswith("gpt-5")

@app.get("/health")
def health():
    return {"status": "ok", "profiles": len(VENUE_PROFILES), "model": OPENAI_MODEL}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        if not raw:
            raise HTTPException(400, "Empty file.")
        b64 = base64.b64encode(raw).decode("utf-8")
    finally:
        await image.close()

    # Quick merchant probe
    quick_prompt = """
Return ONLY this JSON:
{"m": "merchant name or null", "a": "merchant address or null"}
"""
    quick_params = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Read the image and return merchant + address only as JSON."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]},
            {"role": "user", "content": quick_prompt}
        ]
    }
    if supports_temperature(OPENAI_MODEL):
        quick_params["temperature"] = 0.0

    quick = client.chat.completions.create(**quick_params)
    try:
        ma = json.loads(quick.choices[0].message.content or "{}")
        merchant_guess = (ma.get("m") or "")[:200]
        addr_guess = (ma.get("a") or "")[:200]
    except:
        merchant_guess, addr_guess = "", ""

    profile = find_best_profile(VENUE_PROFILES, merchant_guess, addr_guess)
    sys_prompt = build_system_prompt(profile)

    # Main extraction + AI-based scoring
    main_params = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}
        ]
    }
    if supports_temperature(OPENAI_MODEL):
        main_params["temperature"] = 0.1

    resp = client.chat.completions.create(**main_params)

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
                "reason": f"Model returned invalid JSON: {str(e)}"
            }
        }

    # Post-process for nulls/dates
    d = data.get("data", {})
    for k in ("MerchantName", "MerchantAddress", "TransactionDate", "StoreID", "InvoiceId", "CR", "TaxID"):
        d[k] = coerce_nullish(d.get(k))
    d["TransactionDate"] = norm_date(d.get("TransactionDate"))

    data["data"] = d
    return AnalyzeResponse(**data)
