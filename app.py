# app.py — single-call, prompt from ENV (string or file), fallback to prompts.py

import os
import json
import base64
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

try:
    from prompts import SYSTEM_PROMPT as FALLBACK_SYSTEM_PROMPT
except Exception:
    FALLBACK_SYSTEM_PROMPT = "You are a Professional Receipt & Invoice Analyzer."

# -----------------------
# ENV / Client
# -----------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment or .env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Prompt can be provided either as text or as a path to a file
OPENAI_SYSTEM_PROMPT = os.getenv("OPENAI_SYSTEM_PROMPT")               # raw text
OPENAI_SYSTEM_PROMPT_PATH = os.getenv("OPENAI_SYSTEM_PROMPT_PATH")     # path to file

def _resolve_system_prompt() -> str:
    # 1) file path takes priority if provided and readable
    if OPENAI_SYSTEM_PROMPT_PATH and os.path.exists(OPENAI_SYSTEM_PROMPT_PATH):
        try:
            with open(OPENAI_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    # 2) raw env string
    if OPENAI_SYSTEM_PROMPT and OPENAI_SYSTEM_PROMPT.strip():
        return OPENAI_SYSTEM_PROMPT
    # 3) fallback to prompts.py
    return FALLBACK_SYSTEM_PROMPT

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="Scan Invoice API",
    version="0.3.0",
    docs_url="/docs",      # Keep Swagger at /docs
    redoc_url="/redoc"     # Keep ReDoc if you want (optional)
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")  # Redirect works now


# -----------------------
# Schemas
# -----------------------
class AnalyzeResponse(BaseModel):
    data: Dict[str, Any]

# -----------------------
# Helpers
# -----------------------
NULL_STRINGS = {"", "null", "none", "nil", "n/a", "na", "—", "-", "غير متوفر", "غير موجود"}

def clamp_int(v, lo=0, hi=100):
    try:
        i = int(round(float(v)))
        return max(lo, min(hi, i))
    except:
        return 0

def coerce_number(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x)
        s = s.replace(",", "").replace("SAR", "").strip()
        trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        s = s.translate(trans)
        return float(s)
    except:
        return None

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
        "%d-%m-%Y %H:%M", "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
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

def validate_and_score(payload: Dict[str, Any]) -> Dict[str, Any]:
    d = payload.get("data", {})
    subtotal = coerce_number(d.get("Subtotal"))
    tax = coerce_number(d.get("Tax"))
    total = coerce_number(d.get("Total"))

    fraud = 0
    conf = 30
    reasons = []

    if subtotal is not None and tax is not None and total is not None:
        if abs((subtotal + tax) - total) <= 1.0:
            conf += 15
        else:
            fraud += 20
            reasons.append("Subtotal + Tax != Total beyond tolerance.")

    if subtotal is not None and tax is not None:
        expected_tax = subtotal * 0.15
        if abs(expected_tax - tax) <= max(1.0, 0.015 * subtotal):
            conf += 15
        else:
            fraud += 15
            reasons.append("VAT not ~15% of Subtotal.")

    d["fraudScore"] = clamp_int(d.get("fraudScore", fraud))
    d["confidentScore"] = clamp_int(d.get("confidentScore", conf))
    if not d.get("reason"):
        d["reason"] = "; ".join(reasons) if reasons else "Checks passed."
    return {"data": d}

def _build_openai_kwargs(b64_image: str) -> Dict[str, Any]:
    return {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _resolve_system_prompt()},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]},
        ],
        # no temperature here to avoid "unsupported_value" on some models
    }

# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": OPENAI_MODEL,
        "system_prompt": "ENV:PATH" if OPENAI_SYSTEM_PROMPT_PATH else ("ENV:STRING" if OPENAI_SYSTEM_PROMPT else "prompts.py"),
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        if not raw:
            raise HTTPException(400, "Empty file.")
        b64 = base64.b64encode(raw).decode("utf-8")
    finally:
        await image.close()

    # === Single OpenAI call ===
    try:
        resp = client.chat.completions.create(**_build_openai_kwargs(b64))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OpenAI call failed: {e}")

    raw_txt = resp.choices[0].message.content or ""
    try:
        data = json.loads(raw_txt)
        if "data" not in data:
            raise ValueError("Missing 'data' root.")
    except Exception as e:
        data = {
            "data": {
                "MerchantName": None, "MerchantAddress": None, "TransactionDate": None,
                "StoreID": None, "InvoiceId": None, "CR": None, "TaxID": None,
                "Subtotal": None, "Tax": None, "Total": None,
                "fraudScore": 0, "confidentScore": 0,
                "reason": f"Model returned non-JSON or invalid format. {str(e)}"
            }
        }

    # Post-processing
    d = data.get("data", {})
    for k in ("MerchantName","MerchantAddress","TransactionDate","StoreID","InvoiceId","CR","TaxID"):
        d[k] = coerce_nullish(d.get(k))
    d["TransactionDate"] = norm_date(d.get("TransactionDate"))
    data["data"] = d

    return AnalyzeResponse(**validate_and_score(data))
