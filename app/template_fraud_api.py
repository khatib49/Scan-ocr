"""
Standalone Template Fraud Detection API

Exposes /template-fraud/check — accepts an invoice/receipt image and returns
a forensic analysis of whether the document looks like an authentic POS receipt
or a fabricated Word/design-software printout that was printed and scanned to
bypass OCR-based fraud checks.

Optionally accepts a merchantId so the model can compare against the expected
visual fingerprint stored in the venue's TemplateHints profile field.
"""

import json
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.gemini_client import call_gemini_with_image
from app.security import verify_api_key, _mongo_db as DB

import uuid

router = APIRouter(prefix="/template-fraud", tags=["Template Fraud Detection"])

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _data_dir() -> str:
    return os.getenv("DATA_DIR", "data")


def _read_prompt_template() -> str:
    path = os.path.join(_data_dir(), "template_fraud_prompt.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(f"Template fraud prompt file not found at: {path}")


def _build_profile_hints(profile: Optional[Dict[str, Any]]) -> str:
    if not profile:
        return ""
    hints: Dict[str, Any] = profile.get("TemplateHints") or {}
    if not hints:
        return ""
    parts = ["EXPECTED RECEIPT FORMAT for this merchant (use when scoring):"]
    if hints.get("receipt_format"):
        parts.append(f"- Receipt format: {hints['receipt_format']}")
    if hints.get("has_qr_code") is not None:
        parts.append(f"- Should have QR code: {hints['has_qr_code']}")
    if hints.get("has_logo") is not None:
        parts.append(f"- Should have logo: {hints['has_logo']}")
    if hints.get("paper_type"):
        parts.append(f"- Expected paper type: {hints['paper_type']}")
    if hints.get("font_style"):
        parts.append(f"- Expected font style: {hints['font_style']}")
    if hints.get("description"):
        parts.append(f"- Visual description: {hints['description']}")
    return "\n".join(parts)


async def _fetch_profile(merchant_id: int) -> Optional[Dict[str, Any]]:
    """Load a venue profile from MongoDB by MerchantId."""
    try:
        doc = await DB["VenueProfile"].find_one({"profile.MerchantId": merchant_id})
        if doc:
            d = doc.get("profile") or doc
            d.pop("_id", None)
            return d
        # Fallback: top-level MerchantId
        doc = await DB["VenueProfile"].find_one({"MerchantId": merchant_id})
        if doc:
            doc.pop("_id", None)
            return doc
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class TemplateFraudResponse(BaseModel):
    templateFraudScore: Optional[int]
    templateFraudWarning: bool
    templateFraudReasons: list
    templateAuthenticitySignals: list
    templateDocumentType: str
    templatePaperType: str
    templateLayoutType: str
    templateFontType: str
    templateHasQR: Optional[bool]
    isManipulated: bool
    manipulationIndicators: list
    isBlurred: bool
    blurRegions: list
    templateCheckConfidence: str
    merchantId: Optional[int]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

_WARNING_THRESHOLD = 60


@router.post(
    "/check",
    response_model=TemplateFraudResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Check whether a receipt image is a fabricated document (Word/design-software printout)",
)
async def check_template_fraud_endpoint(
    image: UploadFile = File(..., description="Receipt or invoice image to analyze"),
    merchantId: Optional[int] = Form(
        None,
        description=(
            "Optional MerchantId. When provided the detector loads the venue's "
            "TemplateHints from the profile and uses those as expected-format hints."
        ),
    ),
):
    """
    Analyzes the visual/structural characteristics of the uploaded image to
    determine whether it is an authentic POS receipt or a fabricated document:

    - **templateFraudScore** 0–100: higher = more likely fabricated
    - **templateFraudWarning**: True when score ≥ 60
    - **templateDocumentType**: e.g. `thermal_receipt`, `word_document`, `spreadsheet_printout`
    - **templateFraudReasons**: list of detected fraud indicators
    - **templateAuthenticitySignals**: list of positive authenticity signals
    """
    request_id = "tmpl-" + str(uuid.uuid4())

    try:
        raw = await image.read()
        if not raw:
            raise HTTPException(400, "Empty file.")
        content_type = image.content_type or "image/jpeg"
    finally:
        await image.close()

    # Optionally enrich with profile TemplateHints
    profile: Optional[Dict[str, Any]] = None
    if merchantId is not None:
        profile = await _fetch_profile(merchantId)

    prompt_template = _read_prompt_template()
    profile_hints = _build_profile_hints(profile)
    prompt = prompt_template.replace("{{PROFILE_HINTS}}", profile_hints)

    try:
        response_text, _ = await call_gemini_with_image(
            prompt=prompt,
            image_bytes=raw,
            mime_type=content_type,
            temp=0.0,
            request_id=request_id,
            call_type="template_fraud",
        )
        result: Dict[str, Any] = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(502, f"Model returned non-JSON response: {exc}")
    except RuntimeError as exc:
        if "RATE_LIMIT_EXCEEDED" in str(exc):
            raise HTTPException(
                429, "Gemini rate limit exceeded. Please retry shortly."
            )
        raise HTTPException(502, f"Model call failed: {exc}")

    # Validate and clamp score
    raw_score = result.get("templateFraudScore")
    try:
        score: Optional[int] = max(0, min(100, int(raw_score)))
    except (TypeError, ValueError):
        score = None

    return TemplateFraudResponse(
        templateFraudScore=score,
        templateFraudWarning=score is not None and score >= _WARNING_THRESHOLD,
        templateFraudReasons=result.get("fraud_indicators") or [],
        templateAuthenticitySignals=result.get("authenticity_signals") or [],
        templateDocumentType=result.get("document_type_detected", "unknown"),
        templatePaperType=result.get("paper_type", "unknown"),
        templateLayoutType=result.get("layout_type", "unknown"),
        templateFontType=result.get("font_type", "unknown"),
        templateHasQR=result.get("has_qr_code"),
        isManipulated=bool(result.get("is_manipulated", False)),
        manipulationIndicators=result.get("manipulation_indicators") or [],
        isBlurred=bool(result.get("is_blurred", False)),
        blurRegions=result.get("blur_regions") or [],
        templateCheckConfidence=result.get("confidence", "low"),
        merchantId=merchantId,
    )
