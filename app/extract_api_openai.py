from decimal import Decimal
import os, re, json, base64
from typing import Any, Dict, Optional, List
from fastapi import File, Form, Security, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi import HTTPException
from fastapi import APIRouter

from app.security import verify_admin_key

router = APIRouter()

try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("Install OpenAI SDK: pip install openai>=1.0.0")

# ---------- Helpers ----------

def _data_dir() -> str:
    return os.getenv("DATA_DIR", "data")

def _default_image_path() -> str:
    return os.path.join(_data_dir(), "slush.png")

def _read_prompt_template() -> Optional[str]:
    p = os.path.join(_data_dir(), "extract_prompt.txt")
    if not os.path.exists(p):
        return None
    try:
        return open(p, "r", encoding="utf-8").read().strip()
    except Exception:
        return None

def _openai_client() -> "OpenAI":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(500, detail="OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)

def _model_name() -> str:
    return os.getenv("OPENAI_MODELExtractor", "gpt-5")

def _image_file_to_data_url(path: str) -> str:
    if not os.path.exists(path):
        raise HTTPException(400, detail=f"Image file not found: {path}")
    with open(path, "rb") as f:
        raw = f.read()
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _upload_to_data_url(upload: UploadFile) -> str:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(400, detail="Uploaded image is empty")
    mime = upload.content_type or "application/octet-stream"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _inject_merchant_id(prompt_text: str, merchant_id: str) -> str:
    if "{{MERCHANT_ID}}" in prompt_text:
        return prompt_text.replace("{{MERCHANT_ID}}", merchant_id)
    replaced = re.sub(r'("MerchantId"\s*:\s*)(-?\d+)', rf'\1{merchant_id}', prompt_text)
    if replaced != prompt_text:
        return replaced
    return re.sub(r'("MerchantId"\s*:\s*)"(.*?)"', rf'\1"{merchant_id}"', prompt_text)

# ---------- Schema coercion ----------

SCHEMA_TEMPLATE = {
    "MerchantName_Keyword": [],
    "MerchantAddress_Keyword": [],
    "MerchantId": None,
    "Category": "",
    "Sub-category": "",
    "Spending Range (SAR)": "",
    "ExtractionHints": {
        "Language": "Mixed",
        "Total_Label": "",
        "Subtotal_Label": "",
        "Tax_Label": "",
        "CR_Label": None,
        "TaxID_Label": "",
        "Date_Label": "",
        "Time_Label": "",
        "Date_Format": "YYYY/MM/DD",
        "Time_Format": "HH:mm",
        "InvoiceId_Label": "",
        "StoreID_Label": None,
        "MerchantName_Keyword": [],
        "MerchantAddress_Keyword": []
    }
}

def _nz_str(v: Any) -> str:
    return "" if v is None else str(v)

def _as_list_str(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []

def _split_datetime(dt: str) -> tuple[str, str]:
    """Accepts 'YYYY-MM-DD HH:MM' or 'YYYY/MM/DD HH:MM[:SS]' and splits; else returns ("","")."""
    if not dt:
        return "", ""
    s = dt.replace("-", "/").strip()
    parts = s.split()
    if len(parts) == 2:
        date, time = parts
        return date, time
    return "", ""

def _map_alt_shape_to_schema(src: dict, merchant_id: str) -> dict:
    """
    Map model alt payload (like {"data":{MerchantName, MerchantAddress, ...}}) to our schema.
    Drops unknown fields (e.g., fraudScore, confidentScore, reason).
    """
    # unwrap {"data": {...}} if present
    data = src.get("data", src)
    # Build the canonical object
    out = json.loads(json.dumps(SCHEMA_TEMPLATE))  # deep copy

    # Merchant names/addresses as arrays (use the best we have)
    out["MerchantName_Keyword"] = _as_list_str(data.get("MerchantName"))
    out["MerchantAddress_Keyword"] = _as_list_str(data.get("MerchantAddress"))

    # Inject MerchantId from form (never trust model)
    out["MerchantId"] = merchant_id

    # Minimal inference for Category/Sub-category if absent in alt shape
    out["Category"] = _nz_str(data.get("Category"))
    out["Sub-category"] = _nz_str(data.get("Sub-category"))
    out["Spending Range (SAR)"] = _nz_str(data.get("Spending Range (SAR)"))

    # ExtractionHints: map label-bearing fields
    eh = out["ExtractionHints"]
    eh["CR_Label"] = data.get("CR") if data.get("CR") not in ("", None) else None
    eh["TaxID_Label"] = _nz_str(data.get("TaxID"))
    eh["InvoiceId_Label"] = _nz_str(data.get("InvoiceId"))
    eh["StoreID_Label"] = _nz_str(data.get("StoreID")) or None

    # If they gave a combined datetime, split; else try separate
    date_label = _nz_str(data.get("Date_Label")) or _nz_str(data.get("TransactionDate"))
    if date_label and " " in date_label:
        d, t = _split_datetime(date_label)
        eh["Date_Label"], eh["Time_Label"] = d, t
    else:
        eh["Date_Label"] = date_label or _nz_str(data.get("Date"))
        eh["Time_Label"] = _nz_str(data.get("Time"))

    # Labels: if model didn’t provide actual strings from receipt, leave empty
    eh["Total_Label"] = _nz_str(data.get("Total_Label"))
    eh["Subtotal_Label"] = _nz_str(data.get("Subtotal_Label"))
    eh["Tax_Label"] = _nz_str(data.get("Tax_Label"))

    # Mirror arrays
    eh["MerchantName_Keyword"] = out["MerchantName_Keyword"]
    eh["MerchantAddress_Keyword"] = out["MerchantAddress_Keyword"]

    return out

def _coerce_schema(model_json: dict, merchant_id: str) -> dict:
    """
    Accepts either:
    A) Canonical schema (what we want)
    B) Alternate shape (data.MerchantName, TransactionDate, etc.)
    Produces the canonical schema; drops unknown fields.
    """
    # If it already looks canonical, normalize lightly
    if "MerchantName_Keyword" in model_json and "ExtractionHints" in model_json:
        out = json.loads(json.dumps(SCHEMA_TEMPLATE))  # deep copy
        out["MerchantName_Keyword"] = _as_list_str(model_json.get("MerchantName_Keyword"))
        out["MerchantAddress_Keyword"] = _as_list_str(model_json.get("MerchantAddress_Keyword"))
        out["MerchantId"] = merchant_id  # enforce
        out["Category"] = _nz_str(model_json.get("Category"))
        out["Sub-category"] = _nz_str(model_json.get("Sub-category"))
        out["Spending Range (SAR)"] = _nz_str(model_json.get("Spending Range (SAR)"))

        eh_in = model_json.get("ExtractionHints") or {}
        eh_out = out["ExtractionHints"]
        # copy known keys only
        for k in eh_out.keys():
            if k in ("MerchantName_Keyword", "MerchantAddress_Keyword"):
                continue
            if k in eh_in and eh_in[k] not in (None, ""):
                eh_out[k] = eh_in[k]
        # mirrors
        eh_out["MerchantName_Keyword"] = out["MerchantName_Keyword"]
        eh_out["MerchantAddress_Keyword"] = out["MerchantAddress_Keyword"]
        # normalize nullables
        if not eh_out["CR_Label"]:
            eh_out["CR_Label"] = None
        if not eh_out["StoreID_Label"]:
            eh_out["StoreID_Label"] = None
        return out

    # Otherwise, map from the alternate shape
    return _map_alt_shape_to_schema(model_json, merchant_id)
def _dedupe_preserve_order(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for s in seq:
        k = s.strip()
        if not k: continue
        if k not in seen:
            seen.add(k); out.append(k)
    return out

def _ascii_straight_quotes(s: str) -> str:
    return s.replace("’", "'").replace("‘","'").replace("“",'"').replace("”",'"')

def _english_variants(name: str) -> List[str]:
    # only if it looks Latin
    if not re.search(r"[A-Za-z]", name): return [name]
    base = _ascii_straight_quotes(name.strip())
    return _dedupe_preserve_order([base, base.title(), base.upper()])

def _normalize_names_generic(obj: dict) -> None:
    names = obj.get("MerchantName_Keyword") or []
    out: List[str] = []
    for n in names:
        n = (n or "").strip()
        if not n: continue
        out.extend(_english_variants(n))
    obj["MerchantName_Keyword"] = _dedupe_preserve_order(out)

def _normalize_addresses_generic(obj: dict) -> None:
    # Keep as-is, but normalize whitespace and curly quotes; de-dupe
    addrs = [ _ascii_straight_quotes((a or "").strip()) for a in (obj.get("MerchantAddress_Keyword") or []) ]
    obj["MerchantAddress_Keyword"] = _dedupe_preserve_order([a for a in addrs if a])

def _infer_time_format(t: str) -> str:
    t = (t or "").strip()
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", t): return "HH:mm:ss"
    if re.fullmatch(r"\d{1,2}:\d{2}", t):       return "HH:mm"
    return "HH:mm"  # fallback

def _pick_date_format(d: str) -> str:
    d = (d or "").replace("-", "/").strip()
    # very light inference; your schema wants the *format string*, not the value
    if re.fullmatch(r"\d{4}/\d{2}/\d{2}", d): return "YYYY/MM/DD"
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", d): return "DD/MM/YYYY"
    return "YYYY/MM/DD"

def _maybe_parse_total_from_labels(obj: dict) -> Optional[Decimal]:
    # Try to pull a number from the total label (some receipts put amount next to label)
    lbl = obj.get("ExtractionHints",{}).get("Total_Label") or ""
    m = re.search(r"(\d+(?:\.\d{1,2})?)", lbl.replace(",", ""))
    if not m: return None
    try: return Decimal(m.group(1))
    except Exception: return None

def _bucket_spending(total: Optional[Decimal]) -> str:
    if total is None: return ""
    v = float(total)
    if v <= 10:   return "0-10"
    if v <= 50:   return "5-50"
    if v <= 100:  return "50-100"
    if v <= 500:  return "100-500"
    return "500+"

def normalize_generic(obj: dict) -> dict:
    # 1) names and addresses
    _normalize_names_generic(obj)
    _normalize_addresses_generic(obj)

    # 2) fix formats if empty/loose
    eh = obj.get("ExtractionHints", {}) or {}
    if not (eh.get("Time_Format") or "").strip():
        eh["Time_Format"] = _infer_time_format(eh.get("Time_Label",""))
    if not (eh.get("Date_Format") or "").strip():
        eh["Date_Format"] = _pick_date_format(eh.get("Date_Label",""))

    # 3) spending bucket (if model didn’t fill)
    if not (obj.get("Spending Range (SAR)") or "").strip():
        total = _maybe_parse_total_from_labels(obj)
        obj["Spending Range (SAR)"] = _bucket_spending(total)

    # 4) mirrors (ensure hints mirror arrays)
    eh["MerchantName_Keyword"] = obj.get("MerchantName_Keyword") or []
    eh["MerchantAddress_Keyword"] = obj.get("MerchantAddress_Keyword") or []
    obj["ExtractionHints"] = eh
    return obj
# ---------- Endpoint ----------

@router.post(
    "/extract",
    dependencies=[Security(verify_admin_key)],
    summary="Extract receipt JSON in unified schema",
)
async def extract_simple(
    merchantId: str = Form(..., description="Merchant ID to inject"),
    image: Optional[UploadFile] = File(default=None, description="Receipt image")
) -> JSONResponse:

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
    prompt_text = _inject_merchant_id(prompt_text, merchantId.strip())

    if image:
        image_url = _upload_to_data_url(image)
    else:
        image_url = _image_file_to_data_url(_default_image_path())

    client = _openai_client()
    model = _model_name()

    try:
        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return exactly one JSON object that validates against the schema."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        # Chat Completions supports "json_object" here.
        response_format={"type": "json_object"},
        # IMPORTANT: do NOT send temperature/top_p/seed for models that don't support it.
    )
        raw_text = completion.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(502, detail=f"OpenAI error: {e!s}")

    # Print exactly what the model said (may include unwanted fields/wrappers)
    print("\n===== RAW_MODEL_JSON =====")
    print(raw_text)
    merchant_id_int = int(merchantId)
    # Parse model JSON (raw)
    try:
        model_obj = json.loads(raw_text)
    except Exception:
        # If the model didn’t emit valid JSON, show it for debugging
        return PlainTextResponse(content=raw_text, status_code=502, media_type="text/plain")

    # Coerce into your strict schema and inject MerchantId
    coerced = _coerce_schema(model_obj, merchant_id_int)

    # Print what you will actually return (no fraudScore/confidentScore/reason)
    print("\n===== COERCED_JSON =====")
    print(json.dumps(coerced, ensure_ascii=False, indent=2))
    final_obj = normalize_generic(coerced)
    return JSONResponse(content=final_obj, status_code=200)
