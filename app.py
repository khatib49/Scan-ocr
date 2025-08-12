import os, json, base64
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from venueMatcher import load_profiles, find_best_profile

# -----------------------
# ENV / Client
# -----------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment or .env")

# Models
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
PROBE_MODEL  = "gpt-4o-mini"   # cheap vision for probe

# Reasoning / output control
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "minimal")  # minimal|low|medium|high

# Pricing for cost logging (adjust if needed)
PRICE_IN_PER_M = 1.25
PRICE_OUT_PER_M = 10.0

# Prompts / profiles
SYSTEM_PROMPT_PATH   = os.getenv("SYSTEM_PROMPT_PATH", "data/system_prompt.txt")
VENUE_PROFILES_PATH  = os.getenv("VENUE_PROFILES_PATH", "data/venue_profiles.json")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Scan Invoice API", version="0.3.2")

# -----------------------
# Data / schemas
# -----------------------
VENUE_PROFILES = load_profiles(VENUE_PROFILES_PATH)

class AnalyzeResponse(BaseModel):
    data: Dict[str, Any]

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
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        base = f.read().strip()
    # Shorten and remove checklist to reduce reasoning cost
    tail = (
        "\n\n---\n"
        "Do NOT include any checklist. Keep internal reasoning minimal. "
        "Return ONLY the JSON object. "
        'Limit the \"reason\" field to a single short sentence.'
    )
    return base + tail

def build_system_prompt(with_profile: Optional[Dict[str, Any]]) -> str:
    base = load_system_prompt()
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
    return not model_name.lower().startswith("gpt-5")

def log_cost(label, r):
    u = getattr(r, "usage", None)
    if not u:
        return
    in_tok  = getattr(u, "prompt_tokens", 0)
    out_tok = getattr(u, "completion_tokens", 0)
    cost = in_tok * (PRICE_IN_PER_M / 1_000_000) + out_tok * (PRICE_OUT_PER_M / 1_000_000)
    print(f"[{label}] prompt={in_tok}, output={out_tok}, est=${cost:.5f}")

# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "profiles": len(VENUE_PROFILES),
        "model": OPENAI_MODEL,
        "probeModel": PROBE_MODEL
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

    # ---------- Quick merchant probe ----------
    quick_prompt = 'Return ONLY this JSON: {"m": "merchant name or null", "a": "merchant address or null"}'
    quick_params = {
        "model": PROBE_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Read the image and return merchant + address only as JSON."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]},
            {"role": "user", "content": quick_prompt}
        ],
    }
    if supports_temperature(PROBE_MODEL):
        quick_params["temperature"] = 0

    quick = client.chat.completions.create(**quick_params)
    log_cost("probe", quick)
    try:
        ma = json.loads(quick.choices[0].message.content or "{}")
        merchant_guess = (ma.get("m") or "")[:200]
        addr_guess = (ma.get("a") or "")[:200]
    except Exception:
        merchant_guess, addr_guess = "", ""

    profile = find_best_profile(VENUE_PROFILES, merchant_guess, addr_guess)
    sys_prompt = build_system_prompt(profile)

    # ---------- Main extraction (GPT-5, minimal reasoning) ----------
    main_params = {
        "model": OPENAI_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}
        ],
    }
    if supports_temperature(OPENAI_MODEL):
        main_params["temperature"] = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

    # Minimal reasoning setting (safe for GPT-5)
    main_params["metadata"] = {"effort": REASONING_EFFORT}

    resp = client.chat.completions.create(**main_params)
    log_cost("main", resp)

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

    # Post-process
    d = data.get("data", {})
    for k in ("MerchantName", "MerchantAddress", "TransactionDate", "StoreID", "InvoiceId", "CR", "TaxID"):
        d[k] = coerce_nullish(d.get(k))
    d["TransactionDate"] = norm_date(d.get("TransactionDate"))
    data["data"] = d

    return AnalyzeResponse(**data)
