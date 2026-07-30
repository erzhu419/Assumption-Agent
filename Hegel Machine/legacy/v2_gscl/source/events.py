from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .models import stable_hash


@dataclass(frozen=True)
class Event:
    event: str
    stage: str
    trace_id: str
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.payload)
        return {
            "event": self.event,
            "stage": self.stage,
            "trace_id": self.trace_id,
            "payload": payload,
            "payload_hash": stable_hash(payload),
            "event_id": stable_hash(
                {"event": self.event, "stage": self.stage, "trace_id": self.trace_id, "payload": payload}
            )[:24],
            "raw_content_persisted": False,
        }


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class NullEventSink:
    def emit(self, event: Event) -> None:
        return None


class MemoryEventSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def emit(self, event: Event) -> None:
        with self._lock:
            self.events.append(event.to_dict())


class JsonlEventSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: Event) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")
