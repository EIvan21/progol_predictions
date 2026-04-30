"""GCS sync helpers. Imports google-cloud-storage lazily so non-GCP runs don't fail."""
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = os.getenv("GCS_BUCKET", "progol-data-storage")


def _enabled() -> bool:
    return os.getenv("USE_GCS", "false").lower() == "true"


def _client():
    from google.cloud import storage
    return storage.Client()


def upload_file(local_path: Path, gcs_path: str, bucket: str = None) -> None:
    if not _enabled():
        return
    bucket = bucket or _DEFAULT_BUCKET
    blob = _client().bucket(bucket).blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    logger.info(f"uploaded gs://{bucket}/{gcs_path}")


def download_file(gcs_path: str, local_path: Path, bucket: str = None) -> bool:
    if not _enabled():
        return False
    bucket = bucket or _DEFAULT_BUCKET
    blob = _client().bucket(bucket).blob(gcs_path)
    if not blob.exists():
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return True


def upload_dir(local_dir: Path, gcs_prefix: str, bucket: str = None,
               include_suffixes: Iterable[str] = None) -> int:
    if not _enabled() or not local_dir.exists():
        return 0
    bucket = bucket or _DEFAULT_BUCKET
    bkt = _client().bucket(bucket)
    n = 0
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        if include_suffixes and p.suffix not in include_suffixes:
            continue
        rel = p.relative_to(local_dir)
        gcs_path = f"{gcs_prefix.rstrip('/')}/{rel.as_posix()}"
        bkt.blob(gcs_path).upload_from_filename(str(p))
        n += 1
    logger.info(f"uploaded {n} files to gs://{bucket}/{gcs_prefix}")
    return n


def download_dir(gcs_prefix: str, local_dir: Path, bucket: str = None) -> int:
    if not _enabled():
        return 0
    bucket = bucket or _DEFAULT_BUCKET
    local_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for blob in _client().list_blobs(bucket, prefix=gcs_prefix.rstrip("/") + "/"):
        rel = Path(blob.name).relative_to(gcs_prefix.rstrip("/"))
        out = local_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(out))
        n += 1
    return n
