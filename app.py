import os, json, base64, math, re, unicodedata
from typing import Optional, Dict, Any
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from venueMatcher import load_profiles, find_best_profile, normalize_text
from prompts import SYSTEM_PROMPT

# Try to load .env; ok if missing
try:
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment or .env")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Scan Invoice API", version="0.1.1")

# In-memory cache of venue profiles
VENUE_PROFILES = load_profiles(os.getenv("VENUE_PROFILES_PATH", "data/venue_profiles.json"))

class AnalyzeResponse(BaseModel):
    data: Dict[str, Any]

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
        # handle Arabic digits
        trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        s = s.translate(trans)
        return float(s)
    except:
        return None

NULL_STRINGS = {"", "null", "none", "nil", "n/a", "na", "—", "-", "غير متوفر", "غير موجود"}

def coerce_nullish(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    return None if s in NULL_STRINGS else x

def norm_date(s: Optional[str]) -> Optional[str]:
    """Normalize to 'YYYY-MM-DD HH:mm' when possible."""
    if not s:
        return None
    s = str(s).strip().replace("  ", " ")
    # replace Arabic numerals
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    s = s.translate(trans)
    # common patterns
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
    # ISO-like?
    try:
        s2 = s.replace("T", " ").replace("Z", "").split(".")[0]
        dt = datetime.fromisoformat(s2)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return s  # keep as-is if unknown

# Optional QR decode (ZATCA). Safe if lib missing.
try:
    from PIL import Image
    from pyzbar.pyzbar import decode as qr_decode
except Exception:
    Image = None
    qr_decode = None

def decode_zatca_qr(image_bytes: bytes) -> Optional[dict]:
    """Return dict with keys: seller, vat, timestamp, total, vat_amount (when QR present)."""
    if not qr_decode or not Image:
        return None
    try:
        img = Image.open(BytesIO(image_bytes))
        codes = qr_decode(img)
        if not codes:
            return None
        # choose the largest payload
        payload = max(codes, key=lambda c: (c.rect.width * c.rect.height)).data
        if not payload:
            return None
        b = bytes(payload)
        out = {}
        i = 0
        while i + 2 <= len(b):
            tag = b[i]; i += 1
            length = b[i]; i += 1
            val = b[i:i+length]; i += length
            try:
                val_s = val.decode("utf-8", "ignore")
            except:
                val_s = ""
            if tag == 1: out["seller"] = val_s
            elif tag == 2: out["vat"] = val_s
            elif tag == 3: out["timestamp"] = val_s
            elif tag == 4: out["total"] = val_s
            elif tag == 5: out["vat_amount"] = val_s
        # normalize
        if "timestamp" in out and out["timestamp"]:
            out["timestamp"] = norm_date(out["timestamp"])
        for k in ("total", "vat_amount"):
            if k in out and out[k] is not None:
                out[k] = coerce_number(out[k])
        return out
    except Exception:
        return None

def validate_and_score(payload: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    d = payload.get("data", {})
    subtotal = coerce_number(d.get("Subtotal"))
    tax = coerce_number(d.get("Tax"))
    total = coerce_number(d.get("Total"))

    fraud = 0
    conf = 30
    reasons = []

    if profile is None and d.get("MerchantName"):
        fraud += 25
        reasons.append("No venue profile found for a readable merchant name.")

    # Math check
    if subtotal is not None and tax is not None and total is not None:
        if abs((subtotal + tax) - total) <= 1.0:
            conf += 15
        else:
            fraud += 20
            reasons.append("Subtotal + Tax != Total beyond tolerance.")
    # VAT ~15%
    if subtotal is not None and tax is not None:
        expected_tax = subtotal * 0.15
        if abs(expected_tax - tax) <= max(1.0, 0.015 * subtotal):
            conf += 15
        else:
            fraud += 15
            reasons.append("VAT not ~15% of Subtotal.")

    # Profile checks
    if profile:
        mkw = normalize_text(str(profile.get("MerchantName_Keyword", "")))
        akw = normalize_text(str(profile.get("MerchantAddress_Keyword", "")))
        m = normalize_text(str(d.get("MerchantName") or ""))
        a = normalize_text(str(d.get("MerchantAddress") or ""))

        if m and mkw and mkw in m:
            conf += 20
        elif m and mkw:
            fraud += 15
            reasons.append("Merchant keyword does not match profile.")

        if isinstance(profile.get("MerchantAddress_Keyword"), list):
            addr_ok = any(normalize_text(x) in a for x in profile["MerchantAddress_Keyword"] if isinstance(x, str))
        else:
            addr_ok = akw and (akw in a)
        if addr_ok:
            conf += 10
        elif a and akw:
            reasons.append("Address keyword does not match profile.")

        rng = str(profile.get("Spending Range (SAR)", "")).replace("–", "-")
        if "-" in rng and total is not None:
            try:
                lo, hi = rng.split("-")
                lo = float(lo.strip().replace(",", ""))
                hi = float(hi.strip().replace(",", ""))
                if total < lo or total > hi:
                    fraud += 10
                    reasons.append("Total outside venue spending range.")
            except:
                pass

        hints = profile.get("ExtractionHints") or {}
        label_hits = 0
        for k in ("InvoiceId", "StoreID"):
            lab = hints.get(f"{k}_Label")
            if lab and (d.get(k) and str(d[k]).strip()):
                label_hits += 1
        if label_hits:
            conf += 10

    d["fraudScore"] = clamp_int(d.get("fraudScore", fraud))
    d["confidentScore"] = clamp_int(d.get("confidentScore", conf))
    if not d.get("reason"):
        d["reason"] = "; ".join(reasons) if reasons else "Checks passed."
    return {"data": d}

def build_system_prompt(with_profile: Optional[Dict[str, Any]]) -> str:
    base = SYSTEM_PROMPT.strip()
    if with_profile:
        hints = with_profile.get("ExtractionHints") or {}
        slim = {
            "ExtractionHints": {
                k: v for k, v in hints.items()
                if k in {
                    "Language","Total_Label","Subtotal_Label","Tax_Label","CR_Label","TaxID_Label",
                    "Date_Label","Time_Label","Date_Format","Time_Format","InvoiceId_Label","StoreID_Label",
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

    # Quick merchant probe to pick profile
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

    # --- Post-processing hardening: nulls/dates/QR ---
    d = data.get("data", {})
    for k in ("MerchantName","MerchantAddress","TransactionDate","StoreID","InvoiceId","CR","TaxID"):
        d[k] = coerce_nullish(d.get(k))
    d["TransactionDate"] = norm_date(d.get("TransactionDate"))

    # Optional ZATCA QR override (fixes year/values if model slipped)
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
