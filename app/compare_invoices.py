# app/compare_invoices.py
"""
Invoice Comparison & Fraud Detection Engine.

Takes a new invoice image, extracts its data via Gemini, and compares it
against previously stored invoices for the same merchant.  Produces a
composite fraud-risk score inspired by enterprise-grade platforms.
"""

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.gemini_client import call_gemini_with_image
from app.merchant_template_service import get_template
from app.security import _mongo_db as DB, verify_api_key
from app.venue_profiles_api_mongo import find_similar_profile
from app.blob_service import upload_image_bytes

router = APIRouter(tags=["Invoice Comparison"])

# ── Risk thresholds ──────────────────────────────────────────
_RISK_HIGH = 80
_RISK_MEDIUM = 50
_RISK_LOW = 25


def _clean_gemini_json(raw: str) -> dict:
    """Parse JSON from Gemini, stripping markdown fences and fixing common issues."""
    text = raw.strip()
    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Walk through characters and escape control chars inside JSON strings
    fixed: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            fixed.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            fixed.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            fixed.append(ch)
            continue
        # Replace any unescaped control character inside a string value
        if in_string and ord(ch) < 0x20:
            fixed.append(" ")
            continue
        fixed.append(ch)
    return json.loads("".join(fixed))


# ── Schemas ──────────────────────────────────────────────────


class FraudSignal(BaseModel):
    """Individual fraud indicator with its own weight."""

    code: str = Field(description="Machine-readable signal code")
    label: str = Field(description="Human-readable signal name")
    weight: float = Field(description="Contribution to the overall fraud score (0-100)")
    detail: str = Field(description="Explanation of why this signal fired")


class MatchedInvoice(BaseModel):
    invoice_id: Optional[str] = Field(None, description="MongoDB document ID")
    scan_reference: Optional[str] = Field(None, description="Original scan reference")
    similarity_score: float = Field(description="0.0–1.0 similarity to the new invoice")
    is_duplicate: bool = Field(
        description="True if this is a probable duplicate submission"
    )
    differences: List[str] = Field(description="Key field-level differences")
    summary: str = Field(description="One-line analyst explanation")


class TemplateDeviation(BaseModel):
    """A single structural mismatch between the submitted invoice and the merchant's known template."""

    element: str = Field(
        description="Which layout element deviates (e.g. logo_position, totals_section, font_type)"
    )
    expected: str = Field(description="What the merchant template defines")
    observed: str = Field(description="What was found in the submitted invoice")
    severity: Literal["critical", "major", "minor"] = Field(
        description="Impact on fraud score"
    )


class TemplateCheck(BaseModel):
    """Result of comparing the invoice layout against the merchant's structural template."""

    template_found: bool = Field(
        description="Whether a stored template exists for this merchant"
    )
    structural_match_score: float = Field(
        ge=0, le=1, description="0.0 = completely different layout, 1.0 = perfect match"
    )
    deviations: List[TemplateDeviation] = Field(
        description="Specific layout mismatches"
    )
    summary: str = Field(description="One-line structural verdict")


class FraudAssessment(BaseModel):
    """Composite fraud-risk assessment."""

    score: int = Field(
        ge=0, le=100, description="Overall fraud risk score (0 = clean, 100 = fraud)"
    )
    level: Literal["critical", "high", "medium", "low", "clear"] = Field(
        description="Risk classification"
    )
    action: Literal["auto_reject", "manual_review", "flag", "approve"] = Field(
        description="Recommended action"
    )
    signals: List[FraudSignal] = Field(
        description="Individual fraud indicators that contributed to the score"
    )
    explanation: str = Field(description="Plain-language summary for reviewers")


class CompareResponse(BaseModel):
    request_id: str
    timestamp: str
    new_invoice: Dict[str, Any]
    merchant: Dict[str, Any] = Field(description="Matched merchant profile summary")
    stored_invoices_checked: int
    matches: List[MatchedInvoice]
    template_check: TemplateCheck = Field(
        description="Structural layout comparison against the merchant's known invoice template"
    )
    fraud_assessment: FraudAssessment
    verdict: Literal[
        "DUPLICATE_CONFIRMED",
        "DUPLICATE_SUSPECTED",
        "ANOMALY_DETECTED",
        "CLEAR",
        "NO_HISTORY",
    ]


# ── Helpers ──────────────────────────────────────────────────

_EXTRACT_PROMPT = """Extract the following fields from this receipt/invoice image.
Return ONLY a raw JSON object (no markdown, no code fences):
{
  "MerchantName": "string|null",
  "MerchantAddress": "string|null",
  "TransactionDate": "YYYY-MM-DD HH:MM|null",
  "InvoiceId": "string|null",
  "StoreID": "string|null",
  "TaxID": "string|null",
  "CR": "string|null",
  "Subtotal": number|null,
  "Tax": number|null,
  "Total": number|null,
  "Discount": number|null
}
Rules:
- Extract ONLY what is visible on the image.
- Numbers as JSON numbers, not strings; null for missing fields.
- TransactionDate in YYYY-MM-DD HH:MM (24h) format.
"""


def _build_compare_prompt(
    new_invoice: Dict[str, Any],
    stored_invoices: List[Dict[str, Any]],
) -> str:
    return f"""You are a senior invoice fraud analyst at a fintech company.
Your task is to compare a NEW invoice submission against STORED invoices
from the same merchant and produce a detailed fraud assessment.

ANALYSIS FRAMEWORK:
1. DUPLICATE DETECTION — exact or near-exact resubmission of a past transaction.
2. AMOUNT ANOMALY — total/subtotal deviates significantly from the merchant's typical range.
3. VELOCITY CHECK — multiple submissions in an unusually short time window.
4. FIELD TAMPERING — key fields (date, total, invoice ID) inconsistent with merchant patterns.

DUPLICATE CRITERIA (flag as duplicate if ALL match within tolerance):
- Same merchant
- Same or very close date/time (within 5 minutes)
- Same total amount (within ±1 SAR)
- Same or missing invoice ID (matching IDs = strong signal)

AMOUNT ANOMALY:
- Compare the new invoice total against the stored invoices' totals.
- If the new total is >3× the average or <0.2× the average, flag it.

VELOCITY:
- If 3+ invoices share the same date, flag frequency anomaly.

NEW INVOICE:
{json.dumps(new_invoice, ensure_ascii=False, indent=2)}

STORED INVOICES (most recent first):
{json.dumps(stored_invoices, ensure_ascii=False, indent=2)}

Return ONLY a raw JSON object (no markdown, no code fences):
{{
  "comparisons": [
    {{
      "stored_index": 0,
      "is_duplicate": true/false,
      "similarity_score": 0.0-1.0,
      "differences": ["field-level differences"],
      "summary": "one-line explanation"
    }}
  ],
  "fraud_signals": [
    {{
      "code": "DUPLICATE_EXACT|DUPLICATE_NEAR|AMOUNT_ANOMALY|VELOCITY_SPIKE|FIELD_TAMPER|INVOICE_ID_REUSE",
      "label": "Human-readable name",
      "weight": 0-100,
      "detail": "Why this signal fired"
    }}
  ],
  "overall_fraud_score": 0-100,
  "explanation": "2-3 sentence summary for a human reviewer"
}}

SCORING GUIDE:
- 0-24: Clear — no fraud indicators
- 25-49: Low risk — minor anomalies, approve with note
- 50-79: Medium risk — requires manual review
- 80-100: High/critical risk — likely fraud, recommend rejection

If no fraud signals are found, return an empty fraud_signals array and score 0.
"""


def _build_template_check_prompt(
    merchant_template: Dict[str, Any],
) -> str:
    """Build a prompt that asks Gemini to compare the invoice image
    against the merchant's known structural template."""
    # The structural data lives under "InvoiceTemplate" in the stored document
    inner = merchant_template.get("InvoiceTemplate") or merchant_template
    layout_keys = {
        "merchantId",
        "template_name",
        "document_format",
        "header",
        "transaction_info",
        "items_table",
        "totals_section",
        "footer",
        "visual_fingerprint",
    }
    slim_template = {k: v for k, v in inner.items() if k in layout_keys}

    return f"""You are a forensic document analyst. Compare the invoice IMAGE against the merchant's KNOWN TEMPLATE below.
Detect structural deviations indicating forgery or fabrication.

CHECK THESE ELEMENTS:
1. Document format (thermal vs A4, orientation, background)
2. Header (logo, merchant name, address positions)
3. Transaction info (invoice ID, date positions)
4. Items table (style, columns, alignment)
5. Totals section (position, field order)
6. Footer (QR code, messages)
7. Typography (font type, consistency)

SEVERITY: critical (missing/wrong type) | major (wrong position/style) | minor (small difference)

MERCHANT TEMPLATE:
{json.dumps(slim_template, ensure_ascii=False, indent=2)}

RETURN ONLY a JSON object. Keep "expected" and "observed" values under 10 words each.
Report at most 5 deviations (prioritize critical, then major).
{{
  "structural_match_score": 0.0-1.0,
  "deviations": [
    {{
      "element": "short_name",
      "expected": "brief expected",
      "observed": "brief observed",
      "severity": "critical|major|minor"
    }}
  ],
  "summary": "One short sentence"
}}
"""


async def _fetch_merchant_template(merchant_id: Any) -> Optional[Dict[str, Any]]:
    """Load the merchant's structural invoice template from MerchantsTemplatesProfile."""
    mid = str(merchant_id)
    return await get_template(mid)


async def _fetch_stored_invoices(
    merchant_id: Any, project_id: Any, limit: int = 20
) -> List[Dict[str, Any]]:
    """Fetch recent stored invoices for the same merchant from MongoDB."""
    query = {
        "matched_profile.MerchantId": merchant_id,
        "project_id": project_id,
    }
    cursor = (
        DB["invoice"]
        .find(
            query,
            {"_id": 1, "final_result.data": 1, "scanReference": 1, "created_at": 1},
        )
        .sort("created_at", -1)
        .limit(limit)
    )
    results = []
    async for doc in cursor:
        data = (doc.get("final_result") or {}).get("data") or {}
        results.append(
            {
                "_id": str(doc["_id"]),
                "scanReference": doc.get("scanReference"),
                "MerchantName": data.get("MerchantName"),
                "TransactionDate": data.get("TransactionDate"),
                "InvoiceId": data.get("InvoiceId"),
                "StoreID": data.get("StoreID"),
                "Subtotal": data.get("Subtotal"),
                "Tax": data.get("Tax"),
                "Total": data.get("Total"),
                "Discount": data.get("Discount"),
            }
        )
    return results


# ── Fraud scoring ────────────────────────────────────────────


def _classify_risk(score: int) -> tuple[str, str]:
    """Return (level, recommended_action) for a given fraud score."""
    if score >= 90:
        return "critical", "auto_reject"
    if score >= _RISK_HIGH:
        return "high", "auto_reject"
    if score >= _RISK_MEDIUM:
        return "medium", "manual_review"
    if score >= _RISK_LOW:
        return "low", "flag"
    return "clear", "approve"


def _determine_verdict(
    has_duplicates: bool,
    highest_sim: float,
    fraud_score: int,
) -> str:
    if has_duplicates and highest_sim >= 0.95:
        return "DUPLICATE_CONFIRMED"
    if has_duplicates:
        return "DUPLICATE_SUSPECTED"
    if fraud_score >= _RISK_MEDIUM:
        return "ANOMALY_DETECTED"
    return "CLEAR"


# ── Endpoint ─────────────────────────────────────────────────


@router.post(
    "/compare",
    response_model=CompareResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Compare invoice & fraud risk assessment",
    description=(
        "Upload an invoice image (or provide a URL), supply the merchantId, "
        "and get a composite fraud-risk score with template structure verification "
        "and duplicate detection against stored transaction history."
    ),
)
async def compare_invoice(
    request: Request,
    merchantId: str = Form(..., description="The merchant ID to compare against"),
    image: UploadFile = File(None, description="Invoice image file to upload"),
    imageUrl: str = Form(
        None, description="URL of the invoice image (used if no file uploaded)"
    ),
):
    request_id = f"compare-{uuid.uuid4()}"
    project_id = request.state.project["_id"]
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Resolve image source: upload or URL ──────────────────
    image_bytes = None
    blob_url = imageUrl
    mime_type = "image/jpeg"

    if image and image.filename:
        try:
            image_bytes = await image.read()
            if not image_bytes:
                raise HTTPException(400, "Uploaded file is empty.")
            mime_type = image.content_type or "image/jpeg"
        finally:
            await image.close()

        # Upload to blob storage so Gemini can access it
        try:
            _, blob_url = await upload_image_bytes(
                data=image_bytes,
                content_type=mime_type,
                preferred_name=image.filename,
                return_sas=True,
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to upload image: {e}")

    if not blob_url and not image_bytes:
        raise HTTPException(400, "Provide either an image file or imageUrl.")

    # ── Step 1: Extract data from the new invoice ────────────
    gemini_kwargs = {
        "prompt": _EXTRACT_PROMPT,
        "mime_type": mime_type,
        "temp": 0.0,
        "request_id": request_id,
        "call_type": "compare_extract",
    }
    if blob_url:
        gemini_kwargs["image_url"] = blob_url
    else:
        gemini_kwargs["image_bytes"] = image_bytes

    try:
        extract_resp, _ = await call_gemini_with_image(**gemini_kwargs)
        new_invoice = _clean_gemini_json(extract_resp)
    except Exception as e:
        raise HTTPException(500, f"Failed to extract invoice data: {e}")

    merchant_name = (new_invoice.get("MerchantName") or "").strip()
    merchant_addr = (new_invoice.get("MerchantAddress") or "").strip()

    # ── Step 2: Load merchant profile by merchantId ──────────
    merchant_id = merchantId.strip()
    profile_doc = (
        await DB["VenueProfile"].find_one({"profile.MerchantId": int(merchant_id)})
        if merchant_id.isdigit()
        else None
    )
    if not profile_doc:
        profile_doc = await DB["VenueProfile"].find_one({"MerchantId": merchant_id})
    if not profile_doc:
        profile_doc = await DB["VenueProfile"].find_one(
            {"profile.MerchantId": merchant_id}
        )

    profile = None
    if profile_doc:
        profile = profile_doc.get("profile") or profile_doc
        profile.pop("_id", None)

    merchant_summary = {
        "merchantId": merchant_id,
        "name": (
            profile.get("MerchantName_Keyword", [merchant_name])
            if profile
            else [merchant_name]
        ),
        "category": (profile.get("Category") if profile else None),
        "spendingRange": (profile.get("Spending Range (SAR)") if profile else None),
    }

    # ── Step 3: Fetch stored invoices + merchant template ───
    stored = await _fetch_stored_invoices(merchant_id, project_id)
    merchant_template = await _fetch_merchant_template(merchant_id)

    if not merchant_template:
        raise HTTPException(
            404,
            detail=f"Merchant {merchant_id} does not have an existing template. "
            "Please create a template before running comparison.",
        )

    # ── Step 3b: Template structure check ────────────────────
    template_check_result: Optional[Dict[str, Any]] = None
    template_signals: List[FraudSignal] = []

    if merchant_template:
        tmpl_prompt = _build_template_check_prompt(merchant_template)
        tmpl_kwargs = {
            "prompt": tmpl_prompt,
            "mime_type": mime_type,
            "temp": 0.0,
            "request_id": request_id,
            "call_type": "compare_template",
        }
        if blob_url:
            tmpl_kwargs["image_url"] = blob_url
        else:
            tmpl_kwargs["image_bytes"] = image_bytes

        tmpl_resp = ""
        try:
            tmpl_resp, _ = await call_gemini_with_image(**tmpl_kwargs)
            template_check_result = _clean_gemini_json(tmpl_resp)
        except Exception as e:
            print(f"[template-check] Error: {e}")
            print(
                f"[template-check] Raw response (len={len(tmpl_resp)}):\n{tmpl_resp[:2500]}"
            )
            template_check_result = None

    # Convert template deviations into fraud signals
    if template_check_result:
        struct_score = template_check_result.get("structural_match_score", 1.0)
        deviations = template_check_result.get("deviations", [])

        severity_weights = {"critical": 30, "major": 15, "minor": 5}
        for dev in deviations:
            sev = dev.get("severity", "minor")
            template_signals.append(
                FraudSignal(
                    code=f"TEMPLATE_{sev.upper()}",
                    label=f"Template {sev} deviation: {dev.get('element', 'unknown')}",
                    weight=severity_weights.get(sev, 5),
                    detail=f"Expected: {dev.get('expected', '?')} → Observed: {dev.get('observed', '?')}",
                )
            )

        # Overall structural mismatch signal
        if struct_score < 0.6:
            template_signals.append(
                FraudSignal(
                    code="TEMPLATE_LAYOUT_MISMATCH",
                    label="Significant layout mismatch",
                    weight=25,
                    detail=f"Structural match score {struct_score:.0%} — invoice layout diverges significantly from merchant's known template.",
                )
            )

    tcheck = TemplateCheck(
        template_found=merchant_template is not None,
        structural_match_score=(
            round(template_check_result.get("structural_match_score", 0.0), 3)
            if template_check_result
            else 0.0
        ),
        deviations=(
            [
                TemplateDeviation(**d)
                for d in (template_check_result or {}).get("deviations", [])
            ]
            if template_check_result
            else []
        ),
        summary=(
            template_check_result.get("summary", "")
            if template_check_result
            else "No template on file for this merchant."
        ),
    )

    if not stored:
        # Even without history, template check may produce signals
        t_score = min(sum(s.weight for s in template_signals), 100)
        t_level, t_action = _classify_risk(int(t_score))
        return CompareResponse(
            request_id=request_id,
            timestamp=ts,
            new_invoice=new_invoice,
            merchant=merchant_summary,
            stored_invoices_checked=0,
            matches=[],
            template_check=tcheck,
            fraud_assessment=FraudAssessment(
                score=int(t_score),
                level=t_level,
                action=t_action,
                signals=template_signals,
                explanation=(
                    "No previous invoices on file. "
                    + (
                        tcheck.summary
                        if merchant_template
                        else "First submission — approved."
                    )
                ),
            ),
            verdict="NO_HISTORY",
        )

    # ── Step 4: Gemini comparison + fraud analysis ───────────
    compare_prompt = _build_compare_prompt(new_invoice, stored)
    cmp_kwargs = {
        "prompt": compare_prompt,
        "mime_type": mime_type,
        "temp": 0.1,
        "request_id": request_id,
        "call_type": "compare_analysis",
    }
    if blob_url:
        cmp_kwargs["image_url"] = blob_url
    else:
        cmp_kwargs["image_bytes"] = image_bytes

    try:
        compare_resp, _ = await call_gemini_with_image(**cmp_kwargs)
        result = _clean_gemini_json(compare_resp)
    except Exception as e:
        raise HTTPException(500, f"Comparison analysis failed: {e}")

    # ── Step 5: Build typed response ─────────────────────────
    matches: List[MatchedInvoice] = []
    has_duplicates = False
    highest_sim = 0.0

    for comp in result.get("comparisons", []):
        idx = comp.get("stored_index", 0)
        doc = stored[idx] if idx < len(stored) else {}
        sim = comp.get("similarity_score", 0.0)
        dup = comp.get("is_duplicate", False)

        if dup:
            has_duplicates = True
        if sim > highest_sim:
            highest_sim = sim

        matches.append(
            MatchedInvoice(
                invoice_id=doc.get("_id"),
                scan_reference=doc.get("scanReference"),
                similarity_score=round(sim, 3),
                is_duplicate=dup,
                differences=comp.get("differences", []),
                summary=comp.get("summary", ""),
            )
        )

    # Fraud signals from Gemini comparison
    signals: List[FraudSignal] = []
    for sig in result.get("fraud_signals", []):
        signals.append(
            FraudSignal(
                code=sig.get("code", "UNKNOWN"),
                label=sig.get("label", "Unknown signal"),
                weight=min(max(sig.get("weight", 0), 0), 100),
                detail=sig.get("detail", ""),
            )
        )

    # Merge template structure signals
    signals.extend(template_signals)

    # Composite score: Gemini base + template deviation weight (capped at 100)
    gemini_score = result.get("overall_fraud_score", 0)
    template_penalty = sum(s.weight for s in template_signals)
    raw_score = gemini_score + template_penalty * 0.4  # template = 40% weight
    fraud_score = min(max(int(raw_score), 0), 100)
    level, action = _classify_risk(fraud_score)
    explanation = result.get("explanation", "")
    if template_signals:
        explanation += f" Template structure: {tcheck.summary}"

    verdict = _determine_verdict(has_duplicates, highest_sim, fraud_score)

    return CompareResponse(
        request_id=request_id,
        timestamp=ts,
        new_invoice=new_invoice,
        merchant=merchant_summary,
        stored_invoices_checked=len(stored),
        matches=matches,
        template_check=tcheck,
        fraud_assessment=FraudAssessment(
            score=fraud_score,
            level=level,
            action=action,
            signals=signals,
            explanation=explanation,
        ),
        verdict=verdict,
    )
