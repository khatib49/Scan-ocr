# app/merchant_templates_profile_mongo.py  (Mongo-backed — thin router)
"""
Router for merchant invoice template CRUD + AI extraction.

All business logic lives in merchant_template_service; this module
handles only HTTP plumbing (request parsing, status codes, auth).
"""

from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Body,
    Security,
    Query,
    UploadFile,
    File,
    Form,
)
from app.security import verify_admin_key, verify_api_key
from app.merchant_template_service import (
    extract_template_from_image,
    get_template,
    upsert_template,
    list_templates,
    delete_template,
)

router = APIRouter(prefix="/merchant-templates", tags=["Merchant Templates"])


# ---------- extraction endpoint ----------


@router.post(
    "/extract",
    dependencies=[Security(verify_admin_key)],
    summary="Extract an invoice template from a receipt image and save it (admin)",
)
async def extract_and_save(
    merchantId: str = Form(
        ..., description="The merchantId to associate this template with"
    ),
    image: UploadFile = File(
        ..., description="Receipt or invoice image to extract the template from"
    ),
    overwrite: bool = Form(
        False, description="If true, overwrites an existing template for this merchant"
    ),
) -> Dict[str, Any]:
    """
    Sends the uploaded image to Gemini to extract the full visual/structural
    template, then persists it to MongoDB.  Returns the saved document.
    """
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Empty image file")
    mime = image.content_type or "image/jpeg"

    # Guard against accidental overwrites
    if not overwrite:
        existing = await get_template(merchantId)
        if existing:
            raise HTTPException(
                409,
                detail=f"A template for merchantId '{merchantId}' already exists. "
                "Set overwrite=true to replace it.",
            )

    try:
        extracted = await extract_template_from_image(raw, mime, merchantId.strip())
    except RuntimeError as exc:
        raise HTTPException(502, f"Extraction failed: {exc}")
    except ValueError as exc:
        raise HTTPException(502, str(exc))

    saved = await upsert_template(merchantId, extracted)
    return {"ok": True, "template": saved}


# ---------- read endpoints ----------


@router.get(
    "",
    dependencies=[Security(verify_admin_key)],
    summary="List all merchant invoice templates (admin)",
)
async def list_all(
    q: Optional[str] = Query(
        None, description="Case-insensitive search in template_name"
    ),
    merchant_id: Optional[str] = Query(None, description="Filter by merchantId"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    count, docs = await list_templates(
        q=q, merchant_id=merchant_id, limit=limit, offset=offset
    )
    return {"count": count, "templates": docs}


@router.get(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Get the invoice template for a merchant (admin)",
)
async def get_one(merchant_id: str) -> Dict[str, Any]:
    doc = await get_template(merchant_id)
    if not doc:
        raise HTTPException(404, detail="Template not found for this merchantId")
    return {"template": doc}


# ---------- write endpoints ----------


@router.post(
    "",
    dependencies=[Security(verify_admin_key)],
    summary="Create a merchant invoice template manually (admin)",
)
async def create_manual(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Create a template by providing the InvoiceTemplate JSON directly."""
    if "merchantId" not in payload:
        raise HTTPException(400, detail="Missing required field: merchantId")

    invoice_template = payload.get("InvoiceTemplate")
    if invoice_template is None:
        raise HTTPException(400, detail="Missing required field: InvoiceTemplate")
    if not isinstance(invoice_template, dict):
        raise HTTPException(422, detail="InvoiceTemplate must be an object")
    if "template_name" not in invoice_template:
        raise HTTPException(422, detail="InvoiceTemplate must include 'template_name'")

    mid = str(payload["merchantId"])
    existing = await get_template(mid)
    if existing:
        raise HTTPException(
            409, detail=f"A template for merchantId '{mid}' already exists"
        )

    saved = await upsert_template(mid, invoice_template)
    return {"ok": True, "created": True, "template": saved}


@router.patch(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Update the invoice template for a merchant (admin)",
)
async def update_one(
    merchant_id: str,
    patch: Dict[str, Any] = Body(..., description="Fields to update (partial)"),
) -> Dict[str, Any]:
    if "merchantId" in patch:
        incoming = str(patch["merchantId"])
        if incoming != str(merchant_id):
            raise HTTPException(
                400, detail="Cannot change merchantId; use the id in the path"
            )
        del patch["merchantId"]

    if "InvoiceTemplate" in patch:
        tmpl = patch["InvoiceTemplate"]
        if not isinstance(tmpl, dict):
            raise HTTPException(422, detail="InvoiceTemplate must be an object")

    existing = await get_template(merchant_id)
    if not existing:
        raise HTTPException(404, detail="Template not found for this merchantId")

    # Merge InvoiceTemplate if provided
    if "InvoiceTemplate" in patch:
        saved = await upsert_template(merchant_id, patch["InvoiceTemplate"])
    else:
        saved = existing

    return {"ok": True, "updated": True, "template": saved}


@router.delete(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Delete the invoice template for a merchant (admin)",
)
async def delete_one(merchant_id: str) -> Dict[str, Any]:
    removed = await delete_template(merchant_id)
    if not removed:
        raise HTTPException(404, detail="Template not found for this merchantId")
    return {"ok": True, "deleted": True, "removed": removed}
