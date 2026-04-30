"""Model versioning. Each train run writes to models/v_<ts>/ and updates latest.json."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def new_version_dir(model_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("v_%Y%m%d_%H%M%S")
    p = model_root / ts
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_latest_pointer(model_root: Path, version_dir: Path,
                         metadata: Optional[dict] = None) -> Path:
    pointer = model_root / "latest.json"
    payload = {
        "version": version_dir.name,
        "path": str(version_dir.relative_to(model_root)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload.update(metadata)
    pointer.write_text(json.dumps(payload, indent=2))
    return pointer


def resolve_latest(model_root: Path) -> Optional[Path]:
    pointer = model_root / "latest.json"
    if not pointer.exists():
        return None
    info = json.loads(pointer.read_text())
    return model_root / info["path"]


def load_latest_metadata(model_root: Path) -> Optional[dict]:
    pointer = model_root / "latest.json"
    if not pointer.exists():
        return None
    return json.loads(pointer.read_text())
