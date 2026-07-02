# app/analyze_v2.py
"""
Flow v2 for /analyze — single-call pipeline.

  1. Download the image ONCE (bytes reused everywhere)
  2. In parallel: screen-photo check + ZATCA QR decode (both off the event loop)
  3. ONE structured extraction call (provider-agnostic, default Gemini 3 Flash)
  4. Venue match on the extracted name/address (same matcher as v1)
  5. Targeted re-ask only if a matched profile's hints could fill null fields
  6. Deterministic fraud scoring (QR cross-check, math, VAT, profile rules)
  7. Gray-zone arbitration by a second model (default Claude), if enabled
  8. Same response shape as v1 (+ qrVerified, arbitrated fields)

Enabled with FLOW_VERSION=2. v1 remains untouched for instant rollback.
"""

import asyncio
import json
import os
from typing import Any, Dict, Optional

import aiohttp

from app.llm_providers import get_extractor, get_arbitrator
from app.venue_profiles_api_mongo import find_similar_profile
from utils.logger import log_error, log_scan_invoice
from utils.qr import decode_zatca_qr
from utils.scoring import deterministic_score
from utils.screen_detector import detect_screen_photo

# ── Config ──────────────────────────────────────────────────────
ARBITRATION_LOW = int(os.getenv("ARBITRATION_LOW", "40"))
ARBITRATION_HIGH = int(os.getenv("ARBITRATION_HIGH", "70"))
ARBITRATION_MIN_CONFIDENCE = int(os.getenv("ARBITRATION_MIN_CONFIDENCE", "60"))
DOWNLOAD_TIMEOUT_S = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT_S", "15"))

_PROMPT_V2_PATH = os.getenv("PROMPT_V2_PATH", "data/prompt_v2.txt")

EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "MerchantName": {"type": "string", "nullable": True},
        "MerchantAddress": {"type": "string", "nullable": True},
        "TransactionDate": {"type": "string", "nullable": True},
        "StoreID": {"type": "string", "nullable": True},
        "InvoiceId": {"type": "string", "nullable": True},
        "CR": {"type": "string", "nullable": True},
        "TaxID": {"type": "string", "nullable": True},
        "Discount": {"type": "number", "nullable": True},
        "Subtotal": {"type": "number", "nullable": True},
        "Tax": {"type": "number", "nullable": True},
        "Total": {"type": "number", "nullable": True},
        "modelFraudEstimate": {"type": "integer"},
        "clarity": {"type": "integer"},
        "reason": {"type": "string", "nullable": True},
    },
    "required": ["modelFraudEstimate", "clarity"],
}

_HINT_KEYS = {
    "Language", "Total_Label", "Subtotal_Label", "Tax_Label", "CR_Label",
    "TaxID_Label", "Date_Label", "Time_Label", "Date_Format", "Time_Format",
    "InvoiceId_Label", "StoreID_Label", "MerchantName_Keyword", "MerchantAddress_Keyword",
}
_REASK_FIELDS = ["TransactionDate", "InvoiceId", "TaxID", "Subtotal", "Tax", "Total"]


def _load_prompt() -> str:
    with open(_PROMPT_V2_PATH, encoding="utf-8") as f:
        return f.read().strip()


async def _download_image(url: str) -> bytes:
    async with aiohttp.ClientSession(
        headers={"User-Agent": "ScanInvoiceAPI/2.0"}
    ) as session:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_S)
        ) as resp:
            resp.raise_for_status()
            return await resp.read()


def _base_payload(image_url: Optional[str]) -> Dict[str, Any]:
    """Field skeleton — identical to v1's response shape."""
    return {
        "MerchantName": None,
        "MerchantAddress": None,
        "image_url": image_url or None,
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
        "reason": "",
        "needsRescan": False,
        "profileMatched": False,
        "matchSignals": None,
        "merchantNameMissing": False,
        "merchantNotSupported": False,
        "screenPhotoWarning": False,
        "screenPhotoScore": None,
        "qrVerified": None,
        "arbitrated": False,
    }


def _apply_screen(payload: Dict[str, Any], screen_result: Optional[Dict[str, Any]]) -> None:
    if screen_result:
        payload["screenPhotoWarning"] = screen_result.get("action") in (
            "auto_reject", "manual_review",
        )
        payload["screenPhotoScore"] = round(float(screen_result.get("score", 0.0)), 1)


def _build_arbitration_prompt(fields: Dict[str, Any], score: Dict[str, Any]) -> str:
    return f"""You are an independent senior invoice-fraud reviewer. Another AI extracted the data below from the attached KSA receipt image, and a rule engine flagged it as borderline. Re-examine the IMAGE yourself.

FIRST EXTRACTION:
{json.dumps({k: fields.get(k) for k in ("MerchantName", "TransactionDate", "InvoiceId", "TaxID", "Subtotal", "Tax", "Total")}, ensure_ascii=False)}

RULE-ENGINE SIGNALS: {score.get("signals")}

Verify independently: (1) do the key values on the image match the extraction, (2) does the receipt look authentic (fonts, alignment, thermal-print artifacts) or fabricated/edited?

Return ONLY raw JSON:
{{
  "agrees_with_extraction": true/false,
  "corrected_fields": {{"Total": number or null, "Tax": number or null, "InvoiceId": "string or null"}},
  "authentic": true/false,
  "fraud_score": 0-100,
  "verdict": "approve" | "reject" | "manual_review",
  "reason": "one sentence"
}}"""


async def _arbitrate(
    image_bytes: bytes,
    mime_type: str,
    fields: Dict[str, Any],
    score: Dict[str, Any],
    request_id: str,
) -> Optional[Dict[str, Any]]:
    arbitrator = get_arbitrator()
    if arbitrator is None:
        return None
    try:
        text, _ = await arbitrator.generate(
            prompt=_build_arbitration_prompt(fields, score),
            image_bytes=image_bytes,
            mime_type=mime_type,
            temp=0.0,
            request_id=request_id,
            call_type="arbitration",
        )
        return json.loads(text)
    except Exception as e:
        print(f"[v2-arbitration] failed (non-fatal): {e}")
        return None


async def run_analyze_v2(
    image_url: str,
    userReference: str,
    scanReference: str,
    skip_screen_check: bool,
    request_id: str,
    project_id: Any,
) -> Dict[str, Any]:
    """Returns the final payload dict: {"data": {...}} — same shape as v1."""

    # ── 1) Download once ────────────────────────────────────────
    try:
        image_bytes = await _download_image(image_url)
    except Exception as e:
        await log_error(image_url, f"Image download failed: {e}", "v2_image_download",
                        userReference, scanReference=scanReference,
                        extra={"request_id": request_id})
        d = _base_payload(image_url)
        d["reason"] = f"Could not download image: {e}"
        d["needsRescan"] = True
        payload = {"data": d}
        await log_scan_invoice(imageUrl=image_url, merchant_guess=None, address_guess=None,
                               profile=None, raw_text=None, userReference=userReference,
                               final_result=payload, project_id=project_id,
                               request_id=request_id, scanReference=scanReference)
        return payload

    mime_type = "image/png" if image_bytes[:8].startswith(b"\x89PNG") else "image/jpeg"

    # ── 2) Screen check + QR decode in parallel, off the loop ───
    screen_task = (
        asyncio.to_thread(detect_screen_photo, image_bytes)
        if not skip_screen_check
        else None
    )
    qr_task = asyncio.to_thread(decode_zatca_qr, image_bytes)
    if screen_task:
        screen_result, qr_data = await asyncio.gather(screen_task, qr_task)
    else:
        screen_result, qr_data = None, await qr_task

    print(f"[v2] screen={screen_result['action'] if screen_result else 'skipped'} "
          f"qr={'yes' if qr_data else 'no'}")

    if screen_result and screen_result.get("action") == "auto_reject":
        await log_error(image_url,
                        f"Screen photo detected: {screen_result.get('details')}",
                        "screen_photo_detected", userReference,
                        scanReference=scanReference,
                        extra={"request_id": request_id, "screen_detection": screen_result})
        d = _base_payload(image_url)
        d["reason"] = (f"Receipt appears to have been photographed from a screen "
                       f"(score: {screen_result['score']:.1f}/100).")
        _apply_screen(d, screen_result)
        payload = {"data": d}
        await log_scan_invoice(imageUrl=image_url, merchant_guess=None, address_guess=None,
                               profile=None, raw_text=None, userReference=userReference,
                               final_result=payload, project_id=project_id,
                               request_id=request_id, scanReference=scanReference)
        return payload

    # ── 3) Single structured extraction ─────────────────────────
    extractor = get_extractor()
    raw_txt: Optional[str] = None
    try:
        raw_txt, _ = await extractor.generate(
            prompt=_load_prompt(),
            image_bytes=image_bytes,
            mime_type=mime_type,
            temp=0.1,
            request_id=request_id,
            call_type="extract_v2",
            response_schema=EXTRACTION_SCHEMA,
        )
        fields: Dict[str, Any] = json.loads(raw_txt)
    except Exception as e:
        await log_error(image_url, f"Extraction failed: {e}", "v2_extraction",
                        userReference, scanReference=scanReference,
                        extra={"request_id": request_id, "raw": (raw_txt or "")[:2000]})
        d = _base_payload(image_url)
        d["reason"] = f"Extraction failed: {e}"
        d["needsRescan"] = True
        _apply_screen(d, screen_result)
        payload = {"data": d}
        await log_scan_invoice(imageUrl=image_url, merchant_guess=None, address_guess=None,
                               profile=None, raw_text=raw_txt, userReference=userReference,
                               final_result=payload, project_id=project_id,
                               request_id=request_id, scanReference=scanReference)
        return payload

    merchant_guess = (fields.get("MerchantName") or "").strip()[:200]
    addr_guess = (fields.get("MerchantAddress") or "").strip()[:200]

    # ── 4) Merchant missing → same rejection as v1 ──────────────
    if not merchant_guess:
        d = _base_payload(image_url)
        d["MerchantAddress"] = addr_guess or None
        d["reason"] = "Merchant name could not be extracted from the receipt image."
        d["needsRescan"] = True
        d["merchantNameMissing"] = True
        _apply_screen(d, screen_result)
        payload = {"data": d}
        await log_scan_invoice(imageUrl=image_url, merchant_guess=None, address_guess=addr_guess,
                               profile=None, raw_text=raw_txt, userReference=userReference,
                               final_result=payload, project_id=project_id,
                               request_id=request_id, scanReference=scanReference)
        return payload

    # ── 5) Venue match (same matcher & confidence gates as v1) ──
    match = await find_similar_profile(merchant_guess, addr_guess)
    matched = match.get("matched")
    profile = match.get("profile")
    signals = match.get("signals", {})
    match_mode = signals.get("match_mode", "unknown")
    name_score = signals.get("name_fuzzy", 0.0)
    addr_score = signals.get("address_fuzzy") or 0.0
    strong = signals.get("contains_strong_token", False)

    print(f"[v2-match] guess='{merchant_guess}' matched={matched} mode={match_mode} "
          f"name={name_score:.2f} addr={addr_score:.2f} strong={strong}")

    can_assign = False
    if matched:
        if match_mode == "strict":
            can_assign = name_score >= 0.75 and addr_score >= 0.70 and strong
        elif match_mode == "name_only":
            can_assign = name_score >= 0.85 and strong

    if not matched or not can_assign:
        d = _base_payload(image_url)
        d["MerchantName"] = merchant_guess
        d["MerchantAddress"] = addr_guess or None
        d["merchantNotSupported"] = True
        d["profileMatched"] = bool(matched)
        d["matchSignals"] = signals or None
        d["reason"] = signals.get("rejection_reason") or (
            f"Match quality insufficient ({match_mode}): name={name_score:.2%}, "
            f"address={addr_score:.2%}, strong_tokens={strong}"
            if matched else "Merchant is not currently supported."
        )
        _apply_screen(d, screen_result)
        payload = {"data": d}
        await log_scan_invoice(imageUrl=image_url, merchant_guess=merchant_guess,
                               address_guess=addr_guess, profile=profile, raw_text=raw_txt,
                               userReference=userReference, final_result=payload,
                               project_id=project_id, request_id=request_id,
                               scanReference=scanReference)
        return payload

    # ── 5b) Targeted re-ask with profile hints (only if useful) ─
    hints = (profile or {}).get("ExtractionHints") or {}
    missing = [k for k in _REASK_FIELDS if fields.get(k) in (None, "", "null")]
    if hints and missing:
        slim_hints = {k: v for k, v in hints.items() if k in _HINT_KEYS and v}
        try:
            reask_prompt = (
                _load_prompt()
                + "\n\n---\nCONTEXT VENUE PROFILE (label hints only; never fabricate):\n"
                + json.dumps(slim_hints, ensure_ascii=False)
                + f"\n\nFocus on these previously-unreadable fields: {missing}"
            )
            reask_txt, _ = await extractor.generate(
                prompt=reask_prompt,
                image_bytes=image_bytes,
                mime_type=mime_type,
                temp=0.0,
                request_id=request_id,
                call_type="extract_v2_hints",
                response_schema=EXTRACTION_SCHEMA,
            )
            refined = json.loads(reask_txt)
            for k in missing:
                if refined.get(k) not in (None, "", "null"):
                    fields[k] = refined[k]
            raw_txt = reask_txt
        except Exception as e:
            print(f"[v2-reask] non-fatal: {e}")

    # ── 6) Deterministic scoring ─────────────────────────────────
    score = deterministic_score(
        fields=fields,
        profile=profile,
        qr=qr_data,
        screen_result=screen_result,
        model_fraud_estimate=fields.get("modelFraudEstimate"),
    )

    # ── 7) Gray-zone arbitration ─────────────────────────────────
    arbitrated = False
    if (ARBITRATION_LOW <= score["fraudScore"] <= ARBITRATION_HIGH
            or score["confidentScore"] < ARBITRATION_MIN_CONFIDENCE):
        verdict = await _arbitrate(image_bytes, mime_type, fields, score, request_id)
        if verdict:
            arbitrated = True
            corrected = verdict.get("corrected_fields") or {}
            for k, v in corrected.items():
                if v not in (None, "", "null") and fields.get(k) in (None, "", "null"):
                    fields[k] = v
            if verdict.get("verdict") == "reject" and not verdict.get("authentic", True):
                score["fraudScore"] = max(score["fraudScore"], 85)
                score["reasons"].append(f"Arbitrator: {verdict.get('reason', 'rejected')}")
            elif verdict.get("verdict") == "approve" and verdict.get("agrees_with_extraction"):
                score["fraudScore"] = min(score["fraudScore"], ARBITRATION_LOW - 1)
                score["confidentScore"] = min(100, score["confidentScore"] + 20)
            else:
                score["reasons"].append(
                    f"Arbitrator requests manual review: {verdict.get('reason', '')}"
                )

    # ── 8) Final payload (v1 shape + new fields) ─────────────────
    d = _base_payload(image_url)
    for k in ("MerchantName", "MerchantAddress", "TransactionDate", "StoreID",
              "InvoiceId", "CR", "TaxID", "Subtotal", "Tax", "Total"):
        d[k] = fields.get(k)
    d["Discount"] = fields.get("Discount")
    d["clarity"] = fields.get("clarity")
    d["fraudScore"] = score["fraudScore"]
    d["confidentScore"] = score["confidentScore"]
    d["reason"] = ", ".join(score["reasons"]) if score["reasons"] else "All checks passed."
    d["needsRescan"] = score["needsRescan"]
    d["profileMatched"] = True
    d["matchSignals"] = {
        "name_fuzzy": name_score,
        "address_fuzzy": addr_score,
        "contains_strong_token": strong,
        "match_mode": match_mode,
        "best_name": signals.get("best_name"),
        "best_address": signals.get("best_address"),
    }
    d["fraudSignals"] = score["signals"]
    d["qrVerified"] = score["qrVerified"]
    d["arbitrated"] = arbitrated
    _apply_screen(d, screen_result)

    mid = (profile.get("MerchantId") or profile.get("MerchantID")
           or profile.get("merchantId")) if isinstance(profile, dict) else None
    if mid is not None and name_score >= 0.85 and strong:
        d["MerchantId"] = mid
    elif mid is not None and match_mode == "strict":
        d["MerchantId"] = mid

    payload = {"data": d}
    await log_scan_invoice(imageUrl=image_url, merchant_guess=merchant_guess,
                           address_guess=addr_guess, profile=profile, raw_text=raw_txt,
                           userReference=userReference, final_result=payload,
                           project_id=project_id, request_id=request_id,
                           scanReference=scanReference)
    return payload
