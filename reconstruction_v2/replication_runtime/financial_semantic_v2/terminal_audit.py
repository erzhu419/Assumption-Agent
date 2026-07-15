from __future__ import annotations

"""Future-only ordered Codex terminal-event auditing.

Legacy evaluation treated every JSON ``error`` record as terminal.  Codex may
emit a recoverable transport ``error`` and subsequently complete the same turn.
This module classifies only an ordered trace with exactly one
``turn.completed`` and no ``turn.failed`` as successful.  It deliberately does
not alter or reinterpret any previously frozen observation.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from assumption_agent.models import stable_hash


CODEX_TERMINAL_EVENT_AUDIT_VERSION = "codex_ordered_terminal_event_audit_v2"

_ERROR_EVENT = "error"
_COMPLETED_EVENT = "turn.completed"
_FAILED_EVENT = "turn.failed"
_RELEVANT_EVENTS = frozenset(
    {_ERROR_EVENT, _COMPLETED_EVENT, _FAILED_EVENT}
)


def _stream_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        if value.is_symlink() or not value.is_file():
            raise ValueError("terminal trace path must be a regular file")
        return value.read_text(encoding="utf-8", errors="replace")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _ordered_json_events(text: str) -> Iterable[tuple[int, Mapping[str, Any]]]:
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping):
            yield line_number, row


@dataclass(frozen=True)
class CodexTerminalTraceAuditV2:
    """Content-free receipt for one ordered Codex JSONL trace audit."""

    valid: bool
    error_type: str | None
    issue_types: tuple[str, ...]
    trace_sha256: str
    stream_count: int
    parsed_json_event_count: int
    relevant_event_count: int
    error_event_count: int
    turn_completed_count: int
    turn_failed_count: int
    terminal_event_count: int
    error_before_terminal_count: int
    error_after_terminal_count: int
    recovered_transient_error: bool

    def safe_payload(self) -> dict[str, Any]:
        return {
            "audit_version": CODEX_TERMINAL_EVENT_AUDIT_VERSION,
            "valid": self.valid,
            "error_type": self.error_type,
            "issue_types": list(self.issue_types),
            "trace_sha256": self.trace_sha256,
            "stream_count": self.stream_count,
            "parsed_json_event_count": self.parsed_json_event_count,
            "relevant_event_count": self.relevant_event_count,
            "error_event_count": self.error_event_count,
            "turn_completed_count": self.turn_completed_count,
            "turn_failed_count": self.turn_failed_count,
            "terminal_event_count": self.terminal_event_count,
            "error_before_terminal_count": self.error_before_terminal_count,
            "error_after_terminal_count": self.error_after_terminal_count,
            "recovered_transient_error": self.recovered_transient_error,
            "raw_trace_persisted": False,
        }

    @property
    def audit_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.safe_payload()
        return {**payload, "audit_hash": self.audit_hash}


def audit_codex_terminal_trace_v2(
    *streams: object,
) -> CodexTerminalTraceAuditV2:
    """Audit ordered JSONL streams without persisting trace content.

    Streams are concatenated in argument order.  Callers should normally pass
    the durable ``codex.txt`` trace as one argument; independently buffered
    stdout and stderr do not provide a trustworthy cross-stream order.
    """

    texts = tuple(_stream_text(value) for value in streams)
    joined = "".join(
        f"{index}:{len(text.encode('utf-8'))}\n{text}"
        for index, text in enumerate(texts)
    )
    trace_sha256 = hashlib.sha256(joined.encode("utf-8")).hexdigest()

    parsed_count = 0
    relevant: list[tuple[int, str]] = []
    global_position = 0
    for text in texts:
        for _, row in _ordered_json_events(text):
            parsed_count += 1
            event_type = str(row.get("type") or "")
            if event_type in _RELEVANT_EVENTS:
                relevant.append((global_position, event_type))
            global_position += 1

    error_positions = [
        position for position, event_type in relevant
        if event_type == _ERROR_EVENT
    ]
    completed_positions = [
        position for position, event_type in relevant
        if event_type == _COMPLETED_EVENT
    ]
    failed_positions = [
        position for position, event_type in relevant
        if event_type == _FAILED_EVENT
    ]
    terminal_positions = sorted(completed_positions + failed_positions)
    first_terminal = terminal_positions[0] if terminal_positions else None
    errors_before = sum(
        first_terminal is not None and position < first_terminal
        for position in error_positions
    )
    errors_after = sum(
        first_terminal is not None and position > first_terminal
        for position in error_positions
    )

    issues: list[str] = []
    if failed_positions:
        issues.append("codex_turn_failed_observed")
    if not completed_positions:
        issues.append("codex_turn_completed_missing")
    if len(terminal_positions) > 1:
        issues.append("codex_multiple_terminal_events")
    if errors_after:
        issues.append("codex_error_after_terminal")

    valid = not issues and len(completed_positions) == 1
    return CodexTerminalTraceAuditV2(
        valid=valid,
        error_type=issues[0] if issues else None,
        issue_types=tuple(issues),
        trace_sha256=trace_sha256,
        stream_count=len(texts),
        parsed_json_event_count=parsed_count,
        relevant_event_count=len(relevant),
        error_event_count=len(error_positions),
        turn_completed_count=len(completed_positions),
        turn_failed_count=len(failed_positions),
        terminal_event_count=len(terminal_positions),
        error_before_terminal_count=errors_before,
        error_after_terminal_count=errors_after,
        recovered_transient_error=valid and bool(errors_before),
    )
