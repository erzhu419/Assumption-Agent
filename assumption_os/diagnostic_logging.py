"""Small JSONL diagnostic logger for metadata-only HLE artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlDiagnosticLogger:
    """Append structured diagnostic events without retaining raw benchmark text."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        payload.setdefault("raw_content_persisted", False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def log_event(logger: JsonlDiagnosticLogger | None, event: dict[str, Any]) -> None:
    if logger is not None:
        logger.write(event)
