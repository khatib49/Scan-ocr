# app/merchant_templates_api.py  (Mongo-backed)
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body, Security, Query
from bson import ObjectId

from app.security import verify_admin_key, _mongo_db as DB

router = APIRouter(prefix="/merchant-templates", tags=["Merchant Templates"])

_templates_lock = asyncio.Lock()
COLL = lambda: DB["MerchantsTemplatesProfile"]


# ---------- helpers ----------

def _serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ObjectId to string and expose as 'id'."""
    doc = dict(doc)
    if "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc


async def _load_all_from_db() -> List[Dict[str, Any]]:
    cur = COLL().find({})
    return [_serialize_doc(d) async for d in cur]


def _validate_invoice_template(template: Any) -> None:
    """
    Light structural validation for the InvoiceTemplate object.
    Raises HTTPException(422) if the payload is malformed.
    """
    if not isinstance(template, dict):
        raise HTTPException(422, detail="InvoiceTemplate must be an object")
    if "template_name" not in template:
        raise HTTPException(422, detail="InvoiceTemplate must include 'template_name'")
    if "fields" in template and not isinstance(template["fields"], list):
        raise HTTPException(422, detail="InvoiceTemplate.fields must be an array")


# ---------- endpoints ----------

@router.get(
    "",
    dependencies=[Security(verify_admin_key)],
    summary="List all merchant invoice templates (admin)",
)
async def list_templates(
    q: Optional[str] = Query(None, description="Case-insensitive search in template_name"),
    merchant_id: Optional[str] = Query(None, description="Filter by merchantId"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    flt: Dict[str, Any] = {}

    if merchant_id is not None:
        # Accept both string and numeric merchantId
        try:
            flt["merchantId"] = {"$in": [merchant_id, int(merchant_id)]}
        except ValueError:
            flt["merchantId"] = merchant_id

    if q:
        flt["InvoiceTemplate.template_name"] = {"$regex": q, "$options": "i"}

    cur = COLL().find(flt).skip(offset).limit(limit)
    docs = [_serialize_doc(d) async for d in cur]
    count = await COLL().count_documents(flt)

    return {"count": count, "templates": docs}


@router.get(
    "/meta",
    dependencies=[Security(verify_admin_key)],
    summary="Collection metadata (admin)",
)
async def get_meta() -> Dict[str, Any]:
    count = await COLL().estimated_document_count()
    return {
        "collection": "MerchantsTemplatesProfile",
        "db": DB.name,
        "estimated_count": count,
    }


@router.get(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Get the invoice template for a merchant (admin)",
)
async def get_template(merchant_id: str) -> Dict[str, Any]:
    cand = [{"merchantId": merchant_id}]
    try:
        cand.append({"merchantId": int(merchant_id)})
    except ValueError:
        pass

    doc = await COLL().find_one({"$or": cand})
    if not doc:
        raise HTTPException(404, detail="Template not found for this merchantId")

    return {"template": _serialize_doc(doc)}


@router.post(
    "",
    dependencies=[Security(verify_admin_key)],
    summary="Create a merchant invoice template (admin)",
)
async def create_template(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    if "merchantId" not in payload:
        raise HTTPException(400, detail="Missing required field: merchantId")

    invoice_template = payload.get("InvoiceTemplate")
    if invoice_template is None:
        raise HTTPException(400, detail="Missing required field: InvoiceTemplate")
    _validate_invoice_template(invoice_template)

    mid = payload["merchantId"]
    cand = [{"merchantId": mid}]
    try:
        cand.append({"merchantId": int(mid)})
    except (ValueError, TypeError):
        pass

    if await COLL().find_one({"$or": cand}):
        raise HTTPException(409, detail=f"A template for merchantId '{mid}' already exists")

    doc = {
        "merchantId": mid,
        "InvoiceTemplate": invoice_template,
        "CreatedAt": datetime.now(timezone.utc),
        "UpdatedAt": datetime.now(timezone.utc),
    }

    res = await COLL().insert_one(doc)
    created = await COLL().find_one({"_id": res.inserted_id})

    return {"ok": True, "created": True, "template": _serialize_doc(created)}


@router.patch(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Update the invoice template for a merchant (admin)",
)
async def update_template(
    merchant_id: str,
    patch: Dict[str, Any] = Body(..., description="Fields to update (partial)"),
) -> Dict[str, Any]:
    # Prevent changing merchantId via patch body
    if "merchantId" in patch:
        incoming = str(patch["merchantId"])
        if incoming != str(merchant_id):
            raise HTTPException(400, detail="Cannot change merchantId; use the id in the path")
        del patch["merchantId"]

    if "InvoiceTemplate" in patch:
        _validate_invoice_template(patch["InvoiceTemplate"])

    patch["UpdatedAt"] = datetime.now(timezone.utc)

    cand = [{"merchantId": merchant_id}]
    try:
        cand.append({"merchantId": int(merchant_id)})
    except ValueError:
        pass

    doc = await COLL().find_one_and_update(
        {"$or": cand},
        {"$set": patch},
        return_document=True,
    )
    if not doc:
        raise HTTPException(404, detail="Template not found for this merchantId")

    return {"ok": True, "updated": True, "template": _serialize_doc(doc)}


@router.delete(
    "/{merchant_id}",
    dependencies=[Security(verify_admin_key)],
    summary="Delete the invoice template for a merchant (admin)",
)
async def delete_template(merchant_id: str) -> Dict[str, Any]:
    cand = [{"merchantId": merchant_id}]
    try:
        cand.append({"merchantId": int(merchant_id)})
    except ValueError:
        pass

    doc = await COLL().find_one({"$or": cand})
    if not doc:
        raise HTTPException(404, detail="Template not found for this merchantId")

    await COLL().delete_one({"_id": doc["_id"]})

    return {"ok": True, "deleted": True, "removed": _serialize_doc(doc)}


# ---------- lookup helper (for use by other modules) ----------

async def get_template_for_merchant(merchant_id: Any) -> Optional[Dict[str, Any]]:
    """
    Returns the InvoiceTemplate dict for the given merchantId, or None if not found.
    Intended for internal use (e.g. fraud detection pipeline).

    Example:
        tmpl = await get_template_for_merchant(807)
        if tmpl:
            fields = tmpl.get("fields", [])
    """
    mid = str(merchant_id)
    cand = [{"merchantId": mid}]
    try:
        cand.append({"merchantId": int(mid)})
    except (ValueError, TypeError):
        pass

    doc = await COLL().find_one({"$or": cand})
    if not doc:
        return None

    return doc.get("InvoiceTemplate")