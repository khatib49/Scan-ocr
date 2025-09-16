import os, json, base64
from time import perf_counter
from typing import Optional, Dict, Any
import uuid
import re

from fastapi import FastAPI, Query, Request, UploadFile, File, HTTPException, Depends, Form, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from app.venue_profiles_api import router as venue_profiles_router  
from app.projects import router as projects_router
from utils.helpers import ensure_project_indexes     
from app.security import _mongo_db as DB

from .venue_matcher import load_profiles, build_name_index, find_best_profile_indexed
from utils.transforms import coerce_number, coerce_nullish, norm_date, validate_and_score  
from utils.logger import append_blob_op, append_llm_call, ensure_telemetry_indexes, finalize_request_log, init_request_log, log_scan_invoice, log_error, ping_mongo_or_raise

from .security import verify_admin_key, verify_api_key, add_cors
from .blob_service import close_blob_clients, init_blob_clients, upload_image_bytes, assert_blob_ready , build_read_url

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

client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=2)
app = FastAPI(title="Scan Invoice API", version="0.2.0")

app.include_router(projects_router)

app.include_router(venue_profiles_router, dependencies=[Depends(verify_admin_key)])
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
    await ensure_telemetry_indexes()
    await init_blob_clients()
    await ensure_project_indexes(DB)
    # ...
    try:
        await assert_blob_ready()
    except Exception as e:
        print("[startup] Azure Blob not ready:", str(e))

@app.on_event("shutdown")
async def _shutdown():
    await close_blob_clients()
    
@app.get("/health")
def health():
    return {"status": "ok", "profiles": len(VENUE_PROFILES)}

@app.post("/analyze", response_model= AnalyzeResponse, dependencies=[Depends(verify_api_key)], summary="Analyze an invoice image")
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    userReference: str = Form(..., description="Your internal user ID or reference"),
    scanReference: str = Form(..., description="Your internal scan reference"),
    save_image: bool = Form(False, description="If true, saves the uploaded image to Azure Blob Storage")
    ):
    request_id = str(uuid.uuid4())
    t0 = perf_counter()
    try:
        await init_request_log(
            request_id=request_id,
            path="/analyze",
            userReference=userReference,
            scanReference=scanReference,
            meta={"save_image": save_image}
        )

        success = True
        blob_url: Optional[str] = None
        blob_name: Optional[str] = None
        raw_txt: Optional[str] = None

        project_id = request.state.project["_id"]
        # 1) Read file
        try:
                raw = await image.read()
                if not raw:
                    raise HTTPException(400, "Empty file.")
                content_type = image.content_type
                file_size = len(raw)
        finally:
                
                # Close right away to keep file handles short-lived
                await image.close()

            # 2) Optionally save to Azure and get SAS URL (timed)
        if save_image:
                try:
                    preferred = None
                    if image.filename and len(image.filename) < 150 and "." in image.filename:
                        preferred = image.filename.replace("\\", "/").split("/")[-1]

                    up_start = perf_counter()
                    blob_name, blob_url = await upload_image_bytes(
                        raw,
                        content_type=content_type,
                        preferred_name=preferred,
                        return_sas=True,
                    )
                    up_ms = (perf_counter() - up_start) * 1000.0
                    await append_blob_op(
                        request_id=request_id,
                        op="upload_image_bytes",
                        duration_ms=up_ms,
                        success=True,
                        meta={
                            "preferred": preferred,
                            "blob_name": blob_name,
                            "content_type": content_type,
                            "size_bytes": file_size
                        }
                    )

                    # belt & suspenders: if no SAS returned, build a read URL (timed)
                    if not blob_url and blob_name:
                        br_start = perf_counter()
                        blob_url = await build_read_url(blob_name)
                        br_ms = (perf_counter() - br_start) * 1000.0
                        await append_blob_op(
                            request_id=request_id,
                            op="build_read_url",
                            duration_ms=br_ms,
                            success=True,
                            meta={"blob_name": blob_name}
                        )

                    if blob_url:
                        print("[blob] SAS:", blob_url)
                    if blob_name:
                        print("[blob] blob_name:", blob_name)

                except Exception as e:
                    # Log failure with timing if available
                    try:
                        # If upload failed before timing capture, still record a failed op
                        await append_blob_op(
                            request_id=request_id,
                            op="upload_image_bytes",
                            duration_ms=0.0,
                            success=False,
                            meta={"error": str(e)}
                        )
                    except Exception:
                        pass
                    await log_error(None, f"Blob upload failed: {str(e)}", "blob_upload", userReference=userReference, scanReference=scanReference, extra={"request_id": request_id})

        print("[log] about to insert image_url:", blob_url)    

            # 3) Quick pass to guess merchant/address (fast + cheap) — TIMED
        quick_prompt_path = os.getenv("QUICK_PROMPT_PATH", "data/quick_prompt.txt")
        try:
            with open(quick_prompt_path, encoding="utf-8") as f:
                    QUICK_PROMPT = f.read()
        except Exception:
            QUICK_PROMPT = '{"instruction":"Return JSON { \\"m\\": \\"<merchant>\\", \\"a\\": \\"<address>\\" } only."}'

        b64 = base64.b64encode(raw).decode("utf-8")
        img_block = {"type":"image_url","image_url":{"url": blob_url}} if blob_url else {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}
        q_start = perf_counter()

        try:
            quick = await client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=0.0,
                    messages=[
                        {"role":"system","content":"Read the image and return merchant + address only as JSON. DO NOT add text."},
                        {"role":"user","content":[img_block]},
                        {"role":"user","content":QUICK_PROMPT}
                    ]
                    )
        except RateLimitError as e:
            retry_after = _retry_after_seconds(e)
            await log_error(
                    blob_url,
                    f"Rate limit on quick call: {e}",
                    "openai_rate_limit_quick",
                    userReference,
                    scanReference=scanReference,
                    extra={"request_id": request_id, "retry_after": retry_after}
                )
            return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "rate_limit",
                            "stage": "quick",
                            "message": "Upstream rate limit from OpenAI. Please retry.",
                            "retry_after": retry_after,
                        }
                    }
                )
                

            
        q_ms = (perf_counter() - q_start) * 1000.0
        # usage extraction (SDK dependent)
        q_usage = getattr(quick, "usage", None)
        try:
            q_usage = q_usage.model_dump() if hasattr(q_usage, "model_dump") else (q_usage or {})
        except Exception:
            q_usage = q_usage or {}

            await append_llm_call(
                request_id=request_id,
                call_type="quick",
                model=OPENAI_MODEL,
                duration_ms=q_ms,
                usage=q_usage
            )

        try:
            ma = json.loads(quick.choices[0].message.content or "{}")
            merchant_guess = (ma.get("m") or "").strip()[:200]
            addr_guess = (ma.get("a") or "").strip()[:200]
        except Exception as e:
            merchant_guess, addr_guess = "", ""
            await log_error(blob_url, str(e), "quick_guess", userReference=userReference, scanReference=scanReference, extra={"raw_response": quick.choices[0].message.content if quick and getattr(quick, "choices", None) else None})

            # 4) Venue match
        match = find_best_profile_indexed(NAME_INDEX, merchant_guess)
        matched = match.get("matched")
        profile = match.get("profile")

        raw_txt = None
        data: Dict[str, Any] = None  # type: ignore

            # 5) If no match, return minimal with high fraud score
        if not merchant_guess or not matched:
            data = {
                    "data": {
                        "MerchantName": merchant_guess or None,
                        "MerchantAddress": addr_guess or None,
                        "Image": blob_url or None,
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
                        "reason": ("Merchant name missing." if not merchant_guess else "No matching venue profile found."),
                        "needsRescan": merchant_guess is None,
                        "profileMatched": bool(matched) if merchant_guess else False
                    }
                }
            final_payload = data
        else:
            sys = build_system_prompt(profile)
            m_start = perf_counter()
        try:
            resp = await client.chat.completions.create(
                        model=OPENAI_MODEL,
                        temperature=0.1,
                        messages=[
                            {"role":"system","content": sys},
                            {"role":"user","content":[img_block]}
                        ]
                    )
        except RateLimitError as e:
            retry_after = _retry_after_seconds(e)
            await log_error(
                        blob_url,
                        f"Rate limit on main call: {e}",
                        "openai_rate_limit_main",
                        userReference,
                    scanReference=scanReference,
                        extra={"request_id": request_id, "retry_after": retry_after}
                    )
            return JSONResponse(
                        status_code=429,
                        content={
                            "error": {
                                "code": "rate_limit",
                                "stage": "main",
                                "message": "Upstream rate limit from OpenAI. Please retry.",
                                "retry_after": retry_after,
                            }
                        }
                    )
        
        m_ms = (perf_counter() - m_start) * 1000.0
        m_usage = getattr(resp, "usage", None)
        try:
            m_usage = m_usage.model_dump() if hasattr(m_usage, "model_dump") else (m_usage or {})
        except Exception:
            m_usage = m_usage or {}

        await append_llm_call(
                    request_id=request_id,
                    call_type="main",
                    model=OPENAI_MODEL,
                    duration_ms=m_ms,
                    usage=m_usage,
                    extra={"profile_id": profile.get("MerchantId") if isinstance(profile, dict) else None}
                )

        raw_txt = resp.choices[0].message.content or ""
        try:
            data = json.loads(raw_txt)
            if "data" not in data:
                        raise ValueError("Missing 'data' root.")
        except Exception as e:
            await log_error(blob_url, str(e), "parse_openai_response", userReference=userReference, scanReference=scanReference, extra={"raw_response": raw_txt}, project_id=project_id)
            data = {
                        "data": {
                            "MerchantName": None,
                            "MerchantAddress": None,
                            "Image": blob_url or None,
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
                            "reason": f"Model returned non-JSON or invalid format. {str(e)}",
                        }
                    }

                # Validate/score via your custom logic
        final_payload = validate_and_score(data, profile, blob_url, merchant_guess, matched)

            # 7) Persist log (SAS URL included if saved)
        await log_scan_invoice(
                imageUrl=blob_url,
                merchant_guess=merchant_guess if 'merchant_guess' in locals() else None,
                address_guess=addr_guess if 'addr_guess' in locals() else None,
                profile=profile if 'profile' in locals() else None,
                raw_text=raw_txt,
                userReference=userReference,
                final_result=final_payload,
            project_id=project_id,
                request_id=request_id,
                scanReference=scanReference
            )

        # Done
        return AnalyzeResponse(**final_payload)

    except Exception as e:
        success = False
        await log_error(blob_url, f"Analyze failed: {e}", "analyze_handler", userReference=userReference, scanReference=scanReference)
        raise
    finally:
        total_ms = (perf_counter() - t0) * 1000.0
        # Summarize a few useful flags
        summary = {
            "had_profile_match": bool(locals().get("matched", False)),
            "blob_saved": bool(blob_url),
            "blob_name": blob_name,
        }
        await finalize_request_log(
            request_id=request_id,
            success=success,
            total_ms=total_ms,
            summary=summary
        )


_retry_secs_re = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)", re.I)

def _retry_after_seconds(err: RateLimitError) -> float:
    # Prefer header if present
    try:
        resp = getattr(err, "response", None)
        if resp and getattr(resp, "headers", None):
            h = resp.headers or {}
            if "retry-after" in h:
                return float(h["retry-after"])
    except Exception:
        pass
    # Fallback: parse message text ("Please try again in 3.45s")
    try:
        m = str(err)
        g = _retry_secs_re.search(m)
        if g:
            return float(g.group(1))
    except Exception:
        pass
    return 3.0
