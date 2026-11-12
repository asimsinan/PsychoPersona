from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def create_fresh_run_dir(output_root: Path, command_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}-{uuid4().hex[:8]}"
    run_dir = output_root / "runs" / command_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
