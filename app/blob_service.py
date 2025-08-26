# app/blob_service.py
import os
import uuid
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from dotenv import load_dotenv
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import ContentSettings
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

try:
    load_dotenv()
except Exception:
    pass

_CONN = (os.getenv("AZURE_STORAGE_CONNECTION_STRING") or "").strip()
_CONTAINER = (os.getenv("AZURE_BLOB_CONTAINER", "invoicefiles") or "").strip().lower()
_SAS_TTL_MINUTES = int(os.getenv("SAS_TTL_MINUTES", "60"))

_service_client: Optional[BlobServiceClient] = None
_slug_re = re.compile(r"[^A-Za-z0-9._-]+")


def _validate_conn_string(conn: str) -> None:
    if not conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")
    if conn.startswith(("'", '"')) or conn.endswith(("'", '"')):
        raise RuntimeError("Remove quotes around AZURE_STORAGE_CONNECTION_STRING.")
    if "DefaultEndpointsProtocol=" not in conn or "AccountName=" not in conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING looks malformed.")
    # We require AccountKey to mint SAS server-side
    if "AccountKey=" not in conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is missing AccountKey (required to generate SAS).")


def _safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "/").split("/")[-1]
    name = _slug_re.sub("-", name)
    return name[:180]


def _guess_ext_from_content_type(ct: Optional[str]) -> str:
    if not ct: return ".bin"
    ct = ct.lower()
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "tiff" in ct or "tif" in ct: return ".tif"
    if "bmp" in ct: return ".bmp"
    return ".bin"


async def _get_client() -> BlobServiceClient:
    global _service_client
    if _service_client is None:
        _validate_conn_string(_CONN)
        _service_client = BlobServiceClient.from_connection_string(_CONN)
    return _service_client


async def _ensure_container():
    svc = await _get_client()
    container = svc.get_container_client(_CONTAINER)
    try:
        await container.create_container()  # private by default
    except ResourceExistsError:
        pass
    return container


def _account_parts():
    parts = dict(kv.split("=", 1) for kv in _CONN.split(";") if "=" in kv)
    account = parts.get("AccountName")
    suffix = parts.get("EndpointSuffix", "core.windows.net")
    key = parts.get("AccountKey")
    if not account or not key:
        raise RuntimeError("AccountName/AccountKey missing in AZURE_STORAGE_CONNECTION_STRING.")
    return account, suffix, key


def _build_read_sas_url(blob_name: str, ttl_minutes: int) -> str:
    account, suffix, key = _account_parts()
    expires = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    sas = generate_blob_sas(
        account_name=account,
        container_name=_CONTAINER,
        blob_name=blob_name,
        permission=BlobSasPermissions(read=True),
        expiry=expires,
        account_key=key,              # ✅ pass the AccountKey
    )
    return f"https://{account}.blob.{suffix}/{_CONTAINER}/{blob_name}?{sas}"


async def upload_image_bytes(
    data: bytes,
    *,
    content_type: Optional[str] = None,
    preferred_name: Optional[str] = None,
    return_sas: bool = True,
    sas_ttl_minutes: Optional[int] = None,
) -> Tuple[str, Optional[str]]:
    """
    Upload bytes to a PRIVATE container. Optionally return a short-lived READ SAS URL.
    Returns: (blob_name, sas_url_or_None)
    """
    container = await _ensure_container()

    ext = _guess_ext_from_content_type(content_type)
    if preferred_name:
        blob_name = _safe_filename(preferred_name)
        if "." not in blob_name:
            blob_name += ext
    else:
        blob_name = uuid.uuid4().hex + ext

    blob = container.get_blob_client(blob_name)
    await blob.upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type or "application/octet-stream"),
    )

    if return_sas:
        ttl = _SAS_TTL_MINUTES if sas_ttl_minutes is None else int(sas_ttl_minutes)
        # If SAS generation ever fails, DO NOT raise—return (blob_name, None) so caller can fallback.
        try:
            return blob_name, _build_read_sas_url(blob_name, ttl)
        except Exception as e:
            print("[blob] SAS generation failed:", e)
            return blob_name, None
    return blob_name, None


async def build_read_url(blob_name: str, ttl_minutes: Optional[int] = None) -> str:
    ttl = _SAS_TTL_MINUTES if ttl_minutes is None else int(ttl_minutes)
    return _build_read_sas_url(blob_name, ttl)


async def assert_blob_ready():
    await _ensure_container()
