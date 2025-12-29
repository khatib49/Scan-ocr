import  json
from time import perf_counter
from typing import List, Optional, Dict, Any
import uuid


from fastapi import FastAPI, Query, Request, UploadFile, File, HTTPException, Depends, Form, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.gemini_client import call_gemini_with_image, SYSTEM_PROMPT
from app.venue_profiles_api import router as venue_profiles_router  
from app.projects import router as projects_router
from app.venue_profiles_api_mongo import find_similar_profile, router as venue_profiles_mongo_router
from app.extract_api_gemini import router as extract_router
from utils.helpers import ensure_project_indexes     
from app.security import _mongo_db as DB

from .venue_matcher import  build_name_index
from utils.transforms import validate_and_score  
from utils.logger import append_blob_op, ensure_telemetry_indexes, finalize_request_log, init_request_log, log_scan_invoice, log_error, ping_mongo_or_raise

from .security import verify_admin_key, verify_api_key, add_cors
from .blob_service import close_blob_clients, init_blob_clients, upload_image_bytes, assert_blob_ready, build_read_url

app = FastAPI(title="Scan Invoice API (Gemini)", version="9.0.0")

# CORS
add_cors(app)

app.include_router(projects_router)
app.include_router(venue_profiles_mongo_router)
app.include_router(venue_profiles_router, dependencies=[Depends(verify_admin_key)])
app.include_router(extract_router)

# Global caches (hot-reloaded by /venue-profiles/reload)
VENUE_PROFILES: List[Dict[str, Any]] = []
NAME_INDEX: Dict[str, Dict[str, Any]] = {}

async def _load_profiles_from_db() -> list[dict]:
    cur = DB["VenueProfile"].find({})
    out = []
    async for d in cur:
        if "_id" in d:
            del d["_id"]
        # unwrap nested "profile" key if it exists
        if "profile" in d and isinstance(d["profile"], dict):
            d = d["profile"]
        out.append(d)
    return out


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
    return {"status": "ok"}

@app.on_event("startup")
async def _startup_checks():
    # Fail fast on Mongo; warn on Azure
    await ping_mongo_or_raise()
    await ensure_telemetry_indexes()
    await init_blob_clients()
    await ensure_project_indexes(DB)
    global VENUE_PROFILES, NAME_INDEX
    VENUE_PROFILES = await _load_profiles_from_db()
    NAME_INDEX = build_name_index(VENUE_PROFILES)
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


@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(verify_api_key)], summary="Analyze an invoice image")
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    userReference: str = Form(..., description="Your internal user ID or reference"),
    scanReference: str = Form(..., description="Your internal scan reference"),
    save_image: bool = Form(False, description="If true, saves the uploaded image to Azure Blob Storage")
):
    request_id = str(uuid.uuid4())
    t0 = perf_counter()
    
    blob_url: Optional[str] = None
    blob_name: Optional[str] = None
    raw_txt: Optional[str] = None
    success = True
    
    try:
        await init_request_log(
            request_id=request_id,
            path="/analyze",
            userReference=userReference,
            scanReference=scanReference,
            meta={"save_image": save_image}
        )

        project_id = request.state.project["_id"]
        
        # 1) Read file
        try:
            raw = await image.read()
            if not raw:
                raise HTTPException(400, "Empty file.")
            content_type = image.content_type or "image/jpeg"
            file_size = len(raw)
        finally:
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

                # If no SAS returned, build a read URL (timed)
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
                try:
                    await append_blob_op(
                        request_id=request_id,
                        op="upload_image_bytes",
                        duration_ms=0.0,
                        success=False,
                        meta={"error": str(e)}
                    )
                except Exception:
                    pass
                await log_error(
                    None, 
                    f"Blob upload failed: {str(e)}", 
                    "blob_upload", 
                    userReference=userReference, 
                    scanReference=scanReference, 
                    extra={"request_id": request_id}
                )

        # 3) Quick pass to guess merchant/address (fast + cheap)
        quick_prompt = """Return ONLY this raw JSON object:
{"m": "merchant name or null", "a": "merchant address or null"}

Formatting rules:
- DO NOT include ```json or any markdown formatting — return raw JSON only, no code fences or explanations.
- The output must start with '{' and end with '}', with no extra characters before or after.

Extraction rules:
- "m" must be the BUSINESS/STORE/BRAND name selling the goods — not a customer, cashier, or staff name.
- Accept brand names even if they appear next to "Customer" if clearly recognizable (e.g., luxury brands like 'Roberto Coin').
- Ignore names near labels such as "Served by", "Cashier", "Salesperson", "Phone", or "Register".
- Prefer names at the top of the receipt or near store/VAT info — but allow brand names from footer if layout confirms they are merchants.
- "a" is the merchant's physical store address if printed. If not available, set it to null."""

        try:
            print("[quick] calling Gemini for merchant/address guess...")
            quick_response, _ = await call_gemini_with_image(
                prompt=quick_prompt,
                image_bytes=raw,
                mime_type=content_type,
                temp=0.0,
                request_id=request_id,
                call_type="quick"
            )
            print(f"[quick] Gemini response: {quick_response}")
        except RuntimeError as e:
            if "RATE_LIMIT_EXCEEDED" in str(e):
                await log_error(
                    blob_url,
                    f"Rate limit on quick call: {e}",
                    "gemini_rate_limit_quick",
                    userReference,
                    scanReference=scanReference,
                    extra={"request_id": request_id}
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "rate_limit",
                            "stage": "quick",
                            "message": "Gemini API rate limit exceeded. Please wait a moment and try again.",
                            "retry_after": 10,
                        }
                    }
                )
            # Re-raise other RuntimeErrors
            await log_error(
                blob_url,
                f"Gemini runtime error on quick call: {e}",
                "gemini_error_quick",
                userReference,
                scanReference=scanReference,
                extra={"request_id": request_id}
            )
            raise HTTPException(500, f"Gemini error on quick call: {e}")
        except Exception as e:
            await log_error(
                blob_url,
                f"Gemini error on quick call: {e}",
                "gemini_error_quick",
                userReference,
                scanReference=scanReference,
                extra={"request_id": request_id}
            )
            raise HTTPException(500, f"Gemini error on quick call: {e}")

        # Parse quick response
        try:
            ma = json.loads(quick_response)
            merchant_guess = (ma.get("m") or "").strip()[:200]
            addr_guess = (ma.get("a") or "").strip()[:200]
        except Exception as e:
            merchant_guess, addr_guess = "", ""
            await log_error(
                blob_url, 
                str(e), 
                "quick_guess", 
                userReference=userReference, 
                scanReference=scanReference, 
                extra={"raw_response": quick_response}
            )

        # 4) Venue match
        print(f"[quick] merchant_guess='{merchant_guess}' addr_guess='{addr_guess}'")
        match = await find_similar_profile(merchant_guess)
        matched = match.get("matched")
        profile = match.get("profile")
        signals = match.get("signals", {})

        print(f"[match] merchant_guess='{merchant_guess}' matched={matched} profile_id={(profile.get('MerchantId') if profile else None)} signals={signals}")

        data: Dict[str, Any] = None  # type: ignore
        sys = None
            
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
            # 6) Build prompt with profile context
            sys = build_system_prompt(profile)
            
            # Replace {{MERCHANT_ID}} placeholder
            merchant_id = None
            if isinstance(profile, dict):
                merchant_id = (profile.get("MerchantId") 
                             or profile.get("MerchantID") 
                             or profile.get("merchantId"))
            
            if merchant_id is not None:
                sys = sys.replace("{{MERCHANT_ID}}", str(merchant_id))
            else:
                sys = sys.replace("{{MERCHANT_ID}}", "null")

            # Main extraction call
            try:
                print("[main] calling Gemini for full extraction...")
                main_response, _ = await call_gemini_with_image(
                    prompt=sys,
                    image_bytes=raw,
                    mime_type=content_type,
                    temp=0.1,
                    request_id=request_id,
                    call_type="main"
                )
                print(f"[main] Gemini response: {main_response}")
            except RuntimeError as e:
                if "RATE_LIMIT_EXCEEDED" in str(e):
                    await log_error(
                        blob_url,
                        f"Rate limit on main call: {e}",
                        "gemini_rate_limit_main",
                        userReference,
                        scanReference=scanReference,
                        extra={"request_id": request_id}
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": {
                                "code": "rate_limit",
                                "stage": "main",
                                "message": "Gemini API rate limit exceeded. Please wait a moment and try again.",
                                "retry_after": 10,
                            }
                        }
                    )
                
                # Re-raise other RuntimeErrors
                await log_error(
                    blob_url,
                    f"Gemini runtime error on main call: {e}",
                    "gemini_error_main",
                    userReference,
                    scanReference=scanReference,
                    extra={"request_id": request_id}
                )
                raise HTTPException(500, f"Gemini error on main call: {e}")
            except Exception as e:
                # Other errors
                await log_error(
                    blob_url,
                    f"Gemini error on main call: {e}",
                    "gemini_error_main",
                    userReference,
                    scanReference=scanReference,
                    extra={"request_id": request_id}
                )
                raise HTTPException(500, f"Gemini error on main call: {e}")

            raw_txt = main_response
            
            # Parse main response
            try:
                data = json.loads(raw_txt)
                if "data" not in data:
                    # Gemini returned the extraction directly, wrap it
                    data = {"data": data}
            except Exception as e:
                await log_error(
                    blob_url, 
                    str(e), 
                    "parse_gemini_response", 
                    userReference=userReference, 
                    scanReference=scanReference, 
                    extra={"raw_response": raw_txt}, 
                    project_id=project_id
                )
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

            # Only set MerchantId IF we truly trust the name match
            mid = None
            if matched and isinstance(profile, dict):
                # Require strong signals to prevent false positives
                if signals.get("name_fuzzy", 0.0) >= 0.85 and signals.get("contains_strong_token", False):
                    mid = (profile.get("MerchantId")
                        or profile.get("MerchantID")
                        or profile.get("merchantId"))

            if mid is not None:
                final_payload["data"]["MerchantId"] = mid

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
        await log_error(
            blob_url, 
            f"Analyze failed: {e}", 
            "analyze_handler", 
            userReference=userReference, 
            scanReference=scanReference
        )
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
            summary=summary,
        )