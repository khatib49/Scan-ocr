import os
import uuid
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from dotenv import load_dotenv

from azure.storage.blob.aio import BlobServiceClient, ContainerClient, BlobClient
from azure.storage.blob import ContentSettings, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import ResourceExistsError

try:
    load_dotenv()
except Exception:
    pass

_CONN = (os.getenv("AZURE_STORAGE_CONNECTION_STRING") or "").strip()
_CONTAINER = (os.getenv("AZURE_BLOB_CONTAINER", "invoicefiles") or "").strip().lower()
_SAS_TTL_MINUTES = int(os.getenv("SAS_TTL_MINUTES", "60"))
# tuneable concurrency for faster uploads (each upload uses up to N parallel chunks)
_MAX_CONCURRENCY = int(os.getenv("AZURE_BLOB_MAX_CONCURRENCY", "4"))

_service_client: Optional[BlobServiceClient] = None
_container_client: Optional[ContainerClient] = None

_slug_re = re.compile(r"[^A-Za-z0-9._-]+")


def _validate_conn_string(conn: str) -> None:
    if not conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")
    if conn.startswith(("'", '"')) or conn.endswith(("'", '"')):
        raise RuntimeError("Remove quotes around AZURE_STORAGE_CONNECTION_STRING.")
    if "DefaultEndpointsProtocol=" not in conn or "AccountName=" not in conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING looks malformed.")
    if "AccountKey=" not in conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING missing AccountKey (required for SAS).")


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
    start  = datetime.now(timezone.utc) - timedelta(minutes=2)
    sas = generate_blob_sas(
        account_name=account,
        container_name=_CONTAINER,
        blob_name=blob_name,
        permission=BlobSasPermissions(read=True),
        start=start,
        expiry=expires,
        account_key=key,
    )
    return f"https://{account}.blob.{suffix}/{_CONTAINER}/{blob_name}?{sas}"


# ---------- Lifecycle (call from FastAPI startup/shutdown) ----------

async def init_blob_clients() -> None:
    """
    Create BlobServiceClient + ContainerClient once per process and ensure container exists.
    Call from FastAPI @app.on_event('startup').
    """
    global _service_client, _container_client
    if _service_client is None:
        _validate_conn_string(_CONN)
        _service_client = BlobServiceClient.from_connection_string(_CONN)
    if _container_client is None:
        _container_client = _service_client.get_container_client(_CONTAINER)
        try:
            await _container_client.create_container()  # private by default
        except ResourceExistsError:
            pass


async def close_blob_clients() -> None:
    """
    Gracefully close underlying aio HTTP session.
    Call from FastAPI @app.on_event('shutdown').
    """
    global _service_client, _container_client
    try:
        if _service_client is not None:
            await _service_client.close()
    finally:
        _service_client = None
        _container_client = None


# ---------- Public helpers you call from your routes ----------

async def upload_image_bytes(
    data: bytes,
    *,
    content_type: Optional[str] = None,
    preferred_name: Optional[str] = None,
    return_sas: bool = True,
    sas_ttl_minutes: Optional[int] = None,
) -> Tuple[str, Optional[str]]:
    """
    Upload bytes to a PRIVATE container (async, parallel chunking).
    Returns: (blob_name, sas_url_or_None)
    """
    if _container_client is None:
        # Safety: if startup hook wasn't called, lazily init once
        await init_blob_clients()

    ext = _guess_ext_from_content_type(content_type)
    if preferred_name:
        blob_name = _safe_filename(preferred_name)
        if "." not in blob_name:
            blob_name += ext
    else:
        blob_name = uuid.uuid4().hex + ext

    blob: BlobClient = _container_client.get_blob_client(blob_name)

    # ✅ async, non-blocking upload; enable parallel chunking via max_concurrency
    await blob.upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type or "application/octet-stream"),
        max_concurrency=max(1, _MAX_CONCURRENCY),
        length=len(data),  # helps pipeline avoid re-reading
    )

    if not return_sas:
        return blob_name, None

    ttl = _SAS_TTL_MINUTES if sas_ttl_minutes is None else int(sas_ttl_minutes)
    try:
        return blob_name, _build_read_sas_url(blob_name, ttl)
    except Exception as e:
        # Don’t fail the request if SAS minting hiccups; caller can fallback to build_read_url()
        print("[blob] SAS generation failed:", e)
        return blob_name, None


async def build_read_url(blob_name: str, ttl_minutes: Optional[int] = None) -> str:
    ttl = _SAS_TTL_MINUTES if ttl_minutes is None else int(ttl_minutes)
    return _build_read_sas_url(blob_name, ttl)


async def assert_blob_ready():
    # kept for compatibility with your existing startup code
    await init_blob_clients()
