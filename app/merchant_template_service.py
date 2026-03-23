"""
Service layer for merchant invoice template extraction and persistence.

Handles:
  - Gemini vision call to extract template structure from a receipt image
  - CRUD operations against the MerchantsTemplatesProfile Mongo collection
  - Reusable helpers consumed by the router (and potentially other modules)
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.gemini_client import call_gemini_with_image
from app.security import _mongo_db as DB

# ---------------------------------------------------------------------------
# Collection accessor
# ---------------------------------------------------------------------------

COLL = lambda: DB["MerchantsTemplatesProfile"]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _data_dir() -> str:
    return os.getenv(
        "DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    )


def _read_prompt_template() -> Optional[str]:
    path = os.path.join(_data_dir(), "template_extract_prompt.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


async def extract_template_from_image(
    image_bytes: bytes,
    mime_type: str,
    merchant_id: str,
) -> Dict[str, Any]:
    """
    Call Gemini vision to extract the full invoice/receipt template
    from *image_bytes* for the given *merchant_id*.

    Returns the parsed JSON dict.
    Raises RuntimeError on Gemini errors, ValueError on bad JSON.
    """
    prompt_text = _read_prompt_template()
    if not prompt_text:
        raise RuntimeError(
            "Template extraction prompt file not found (data/template_extract_prompt.txt)"
        )

    prompt_text = prompt_text.replace("{{MERCHANT_ID}}", merchant_id)

    response_text, _ = await call_gemini_with_image(
        prompt=prompt_text,
        image_bytes=image_bytes,
        mime_type=mime_type,
        temp=0.1,
        call_type="template_extract",
    )

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Mongo _id → string 'id'."""
    doc = dict(doc)
    if "_id" in doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


def _merchant_filter(merchant_id: str) -> Dict[str, Any]:
    """Build a $or filter that matches both string and int merchantId."""
    cands: list = [{"merchantId": merchant_id}]
    try:
        cands.append({"merchantId": int(merchant_id)})
    except (ValueError, TypeError):
        pass
    return {"$or": cands}


async def get_template(merchant_id: str) -> Optional[Dict[str, Any]]:
    """Return the full template document for *merchant_id*, or None."""
    doc = await COLL().find_one(_merchant_filter(merchant_id))
    return _serialize_doc(doc) if doc else None


async def upsert_template(
    merchant_id: str,
    invoice_template: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Insert or replace the InvoiceTemplate for *merchant_id*.
    Returns the saved document (serialised).
    """
    now = datetime.now(timezone.utc)
    existing = await COLL().find_one(_merchant_filter(merchant_id))

    if existing:
        await COLL().update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "InvoiceTemplate": invoice_template,
                    "UpdatedAt": now,
                }
            },
        )
        doc = await COLL().find_one({"_id": existing["_id"]})
    else:
        doc = {
            "merchantId": merchant_id,
            "InvoiceTemplate": invoice_template,
            "CreatedAt": now,
            "UpdatedAt": now,
        }
        res = await COLL().insert_one(doc)
        doc = await COLL().find_one({"_id": res.inserted_id})

    return _serialize_doc(doc)


async def list_templates(
    q: Optional[str] = None,
    merchant_id: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> tuple[int, List[Dict[str, Any]]]:
    """Return (total_count, [docs]) with optional search / filter."""
    flt: Dict[str, Any] = {}

    if merchant_id is not None:
        flt.update(_merchant_filter(merchant_id))

    if q:
        flt["InvoiceTemplate.template_name"] = {"$regex": q, "$options": "i"}

    cur = COLL().find(flt).skip(offset).limit(limit)
    docs = [_serialize_doc(d) async for d in cur]
    count = await COLL().count_documents(flt)
    return count, docs


async def delete_template(merchant_id: str) -> Optional[Dict[str, Any]]:
    """Delete and return the removed document, or None if not found."""
    doc = await COLL().find_one(_merchant_filter(merchant_id))
    if not doc:
        return None
    await COLL().delete_one({"_id": doc["_id"]})
    return _serialize_doc(doc)
