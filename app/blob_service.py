# app/blob_service.py
import os
import uuid
from typing import Optional, Tuple

from dotenv import load_dotenv
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

load_dotenv()

_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "scan-invoice-images").strip().lower()
_PUBLIC_BASE = (os.getenv("AZURE_BLOB_PUBLIC_BASE") or "").rstrip("/")

if not _CONN:
    # We keep it lazy: functions will raise if called without this
    pass

_service_client: Optional[BlobServiceClient] = None

async def _get_client() -> BlobServiceClient:
    global _service_client
    if _service_client is None:
        if not _CONN:
            raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")
        _service_client = BlobServiceClient.from_connection_string(_CONN)
    return _service_client

async def _ensure_container():
    svc = await _get_client()
    container = svc.get_container_client(_CONTAINER)
    try:
        await container.create_container()
    except ResourceExistsError:
        pass
    return container

def _guess_ext_from_content_type(ct: Optional[str]) -> str:
    if not ct:
        return ".bin"
    ct = ct.lower()
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "tiff" in ct or "tif" in ct:
        return ".tif"
    if "bmp" in ct:
        return ".bmp"
    return ".bin"

def _build_public_url(container: str, blob_name: str) -> Optional[str]:
    if not _PUBLIC_BASE:
        return None
    return f"{_PUBLIC_BASE}/{container}/{blob_name}"

async def upload_image_bytes(
    data: bytes,
    content_type: Optional[str] = None,
    preferred_name: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    Upload bytes to Azure Blob Storage.
    Returns: (blob_name, public_url_or_none)
    """
    container = await _ensure_container()
    ext = _guess_ext_from_content_type(content_type)
    blob_name = preferred_name or (uuid.uuid4().hex + ext)
    blob = container.get_blob_client(blob_name)

    await blob.upload_blob(data, overwrite=True, content_type=content_type or "application/octet-stream")

    return blob_name, _build_public_url(_CONTAINER, blob_name)
