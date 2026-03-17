from datetime import datetime
import  json
from time import perf_counter
from typing import List, Optional, Dict, Any
from unittest import signals
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
from utils.screen_detector import detect_screen_photo, THRESHOLD_AUTO_REJECT, THRESHOLD_MANUAL_REVIEW

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
    skip_screen_check: bool = Form(False, description="If true, skips screen capture "),
    save_image: bool = Form(True, description="If true, saves the uploaded image to Azure Blob Storage")
):
    request_id = str(uuid.uuid4())
    t0 = perf_counter()
    
    blob_url: Optional[str] = None
    blob_name: Optional[str] = None
    raw_txt: Optional[str] = None
    success = True
    ai_detection_result: Optional[Dict[str, Any]] = None
    
    try:
        await init_request_log(
            request_id=request_id,
            path="/analyze",
            userReference=userReference,
            scanReference=scanReference,
            meta={}
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

        # 2) Save to Azure Blob Storage FIRST (always save for audit trail)
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

            # If no SAS returned, build a read URL
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
            # Continue even if blob upload fails
            blob_url = None
            blob_name = None

        # 1.5) Screen photo detection — FIRST CHECK before anything else
        screen_result = None
        screen_flagged = False
        if not skip_screen_check:
            screen_result = detect_screen_photo(raw, content_type)
            print(f"[screen-detect] score={screen_result['score']} action={screen_result['action']}")

            screen_flagged = (
                not skip_screen_check and
                screen_result["action"] == "manual_review"
            )

            if screen_result["action"] == "auto_reject":
                # Both high confidence → fraudScore 100, needsRescan true
                await log_error(
                    blob_url,
                    f"Screen photo detected (confidence={screen_result['confidence']}): {screen_result['details']}",
                    "screen_photo_detected",
                    userReference=userReference,
                    scanReference=scanReference,
                    extra={"request_id": request_id, "screen_detection": screen_result}
                )

                final_payload = {
                    "data": {
                        "MerchantName":          None,
                        "MerchantAddress":       None,
                        "image_url":             blob_url or None,
                        "MerchantId":            None,
                        "TransactionDate":       None,
                        "StoreID":               None,
                        "InvoiceId":             None,
                        "CR":                    None,
                        "TaxID":                 None,
                        "Subtotal":              None,
                        "Tax":                   None,
                        "Total":                 None,
                        "fraudScore":            100,
                        "confidentScore":        0,
                        "reason":                f"Receipt appears to have been photographed from a screen (score: {screen_result['score']:.1f}/100).",
                        "needsRescan":           False,
                        "profileMatched":        False,
                        "merchantNameMissing":   False,
                        "merchantNotSupported":  False,
                        "screenPhotoWarning":    True,
                        "screenPhotoScore":      screen_result["score"],
                    }
                }

                await log_scan_invoice(
                    imageUrl=blob_url,
                    merchant_guess=None,
                    address_guess=None,
                    profile=None,
                    raw_text=None,
                    userReference=userReference,
                    final_result=final_payload,
                    project_id=project_id,
                    request_id=request_id,
                    scanReference=scanReference
                )
                return AnalyzeResponse(**final_payload)

        # 4) Quick pass to guess merchant/address (fast + cheap)
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

        # 5) Venue match
        print(f"[quick] merchant_guess='{merchant_guess}' addr_guess='{addr_guess}'")
        match = await find_similar_profile(merchant_guess, addr_guess)
        matched = match.get("matched")
        profile = match.get("profile")
        signals = match.get("signals", {})
        match_mode = signals.get("match_mode", "unknown")

        print(f"[match] merchant_guess='{merchant_guess}' matched={matched} profile_id={(profile.get('MerchantId') if profile else None)} signals={signals}")

        data: Dict[str, Any] = None  # type: ignore
        sys = None
            
        
        # ── 6) Merchant name missing ───────────────────────────────────
        # Gemini could not extract any merchant name from the image at all
        if not merchant_guess:
            data = {
                "data": {
                    "MerchantName":          None,
                    "MerchantAddress":       addr_guess or None,
                    "image_url":                 blob_url or None,
                    "MerchantId":            None,
                    "TransactionDate":       None,
                    "StoreID":               None,
                    "InvoiceId":             None,
                    "CR":                    None,
                    "TaxID":                 None,
                    "Subtotal":              None,
                    "Tax":                   None,
                    "Total":                 None,
                    "fraudScore":            100,
                    "confidentScore":        0,
                    "reason":                "Merchant name could not be extracted from the receipt image.",
                    "needsRescan":           True,
                    "profileMatched":        False,
                    "matchSignals":          None,
                    "merchantNameMissing":   True,   # ← Gemini returned null for merchant name
                    "merchantNotSupported":  False,
                    "screenPhotoWarning":    False,
                    "screenPhotoScore":      None,
                }
            }
            final_payload = data

        # ── 7) Merchant not supported ──────────────────────────────────
        # Name was extracted but no matching profile exists in our system
        elif not matched:
            rejection_reason = signals.get("rejection_reason", "Merchant is not currently supported.")

            data = {
                "data": {
                    "MerchantName":          merchant_guess or None,
                    "MerchantAddress":       addr_guess or None,
                    "image_url":                 blob_url or None,
                    "MerchantId":            None,
                    "TransactionDate":       None,
                    "StoreID":               None,
                    "InvoiceId":             None,
                    "CR":                    None,
                    "TaxID":                 None,
                    "Subtotal":              None,
                    "Tax":                   None,
                    "Total":                 None,
                    "fraudScore":            100,
                    "confidentScore":        0,
                    "reason":                rejection_reason,
                    "needsRescan":           False,
                    "profileMatched":        False,
                    "matchSignals":          signals if signals else None,
                    "merchantNameMissing":   False,
                    "merchantNotSupported":  True,   # ← Name found but not in our supported merchants
                    "screenPhotoWarning":    False,
                    "screenPhotoScore":      None,
                }
            }
            final_payload = data
        else:
            
            # Profile matched - now verify confidence for MerchantId assignment
            
            name_score = signals.get("name_fuzzy", 0.0)
            addr_score = signals.get("address_fuzzy", 0.0)
            has_strong_tokens = signals.get("contains_strong_token", False)
            match_mode = signals.get("match_mode", "unknown")

            # Determine if we can confidently assign MerchantId based on mode
            can_assign_merchant_id = False
            
            if match_mode == "strict":
                # STRICT MODE: Require good name + address scores + strong tokens
                can_assign_merchant_id = (
                    name_score >= 0.75 and 
                    addr_score >= 0.70 and 
                    has_strong_tokens
                )
            elif match_mode == "name_only":
                # NAME-ONLY MODE: Require high name score + strong tokens (already validated by matcher)
                can_assign_merchant_id = (
                    name_score >= 0.85 and 
                    has_strong_tokens
                )
            
            if not can_assign_merchant_id:
                # NOT CONFIDENT ENOUGH - REJECT THE TRANSACTION
                print(f"[match] Match found but not confident enough for MerchantId assignment: mode={match_mode}, name={name_score:.2f}, addr={addr_score:.2f}, strong_tokens={has_strong_tokens}")
                
                # Return rejection response
                data = {
                    "data": {
                        "MerchantName": merchant_guess or None,
                        "MerchantAddress": addr_guess or None,
                        "image_url": blob_url or None,
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
                        "merchantNameMissing":   False,
                        "merchantNotSupported":  True,   # ← Name found but confidence too low to support this merchant
                        "screenPhotoWarning":    False,
                        "screenPhotoScore":      None,
                        "reason": f"Match quality insufficient ({match_mode} mode): name={name_score:.2%}, address={addr_score:.2%}, strong_tokens={has_strong_tokens}",
                        "needsRescan": False,
                        "profileMatched": True,  # Match found but not confident enough
                        "matchSignals": {
                            "name_fuzzy": name_score,
                            "address_fuzzy": addr_score,
                            "contains_strong_token": has_strong_tokens,
                            "match_mode": match_mode,
                            "best_name": signals.get("best_name"),
                            "best_address": signals.get("best_address"),
                            "rejection_reason": "Confidence threshold not met for MerchantId assignment"
                        }
                    }
                }
                final_payload = data
            else:
                # 7) Build prompt with profile context
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
                            "image_url": blob_url or None,
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
                
                final_payload["data"]["merchantNameMissing"] = False
                final_payload["data"]["merchantNotSupported"] = False
                
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
                
        # Apply screen photo flag for medium confidence (50–69)
        if not skip_screen_check:
            final_payload["data"]["screenPhotoWarning"] = screen_flagged
            final_payload["data"]["screenPhotoScore"]   = round(screen_result["score"], 1)
        # 9) Persist log (SAS URL included if saved)
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
            "ai_detection": ai_detection_result if ai_detection_result else None
        }
        await finalize_request_log(
            request_id=request_id,
            success=success,
            total_ms=total_ms,
            summary=summary,
        )

@app.post("/detect", dependencies=[Depends(verify_api_key)], summary="Detect if image is AI-generated ")
async def detect_ai_generated(
    image: UploadFile = File(..., description="The image file to analyze for AI generation"),
) -> Dict[str, Any]:
    """
    First-pass check to detect if image is AI-generated using SynthID verification.
    Returns dict with: {"is_ai_generated": bool, "confidence": str, "details": str}
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    verification_prompt = f"""
    Identify if this image contains a digital watermark or technical metadata signifying it was AI-generated.@synthid

    SCOPE:
    1. : Scan specifically for the SynthID digital watermark.
    2. DIGITAL ARTIFACTS: Check for invisible watermarking or metadata indicators of AI origin.
    3. IGNORE TEXT CONTENT: Do NOT flag the image for spelling errors, regional terms (e.g., 'Siyal'), or dates (the current date is {current_time}). These are valid for this document type.

    Return ONLY this raw JSON object:
    {{
    "ai_generated": true/false,
    "confidence": "high/medium/low",
    "reason": "Identify if a watermark was detected or if no technical signs of AI generation exist."
    }}
    """

    try:
        
        request_id = "ai-detect-" + str(uuid.uuid4())
        try:
            raw = await image.read()
            if not raw:
                raise HTTPException(400, "Empty file.")
            content_type = image.content_type or "image/jpeg"
        finally:
            await image.close()
        print("[ai-detect] calling Gemini for AI generation detection...")
        response, _ = await call_gemini_with_image(
            prompt=verification_prompt,
            image_bytes=raw,
            model_name="gemini-3-flash-preview",
            mime_type=content_type,
            temp=1.0,
            request_id= request_id,
            call_type="ai_detection"
        )
        print(f"[ai-detect] Gemini response: {response}")
        
        # Parse response
        result = json.loads(response)
        return {
            "is_ai_generated": result.get("ai_generated", False),
            "is_digital_fabrication": result.get("digital_fabrication", False),
            "confidence": result.get("confidence", "unclear"),
            "details": result.get("reason", "No details provided")
        }
    except Exception as e:
        print(f"[ai-detect] Error during AI detection: {e}")
        await log_error(
            None,
            f"AI detection failed: {e}",
            "ai_detection_error",
            extra={"request_id": request_id}
        )
        # Return uncertain result on error
        return {
            "is_ai_generated": False,
            "confidence": "unclear",
            "details": f"Detection failed: {str(e)}"
        }

