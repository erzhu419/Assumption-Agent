"""First-party runtime tracing for assumption-bearing agent calls.

The post-hoc manifest logger can ingest existing logs, but self-evolution needs
the live runner to emit structured events at the moment it makes assumption
choices.  This recorder keeps those events compact and redacted, then reuses
``manifest_logger`` to turn them into TrialManifest records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .graph_memory import JsonlGraphStore
from .manifest_logger import build_component_manifest_payload, redact_secrets


@dataclass
class RuntimeTraceRecorder:
    eval_id: str
    events_out: Path | None = None
    summary_out: Path | None = None
    graph_dir: Path | None = None
    writeback: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.events_out or self.summary_out or self.writeback)

    def record(self, **event: Any) -> None:
        if not self.enabled:
            return
        self.events.append(redact_secrets(event))

    def record_llm_call(
        self,
        *,
        problem_id: str,
        component: str,
        prompt_kind: str,
        assumption: str,
        expected_effect: str,
        observed_effect: str,
        why_selected: str = "This runner step requires an LLM inference.",
        status: str = "observed",
        artifacts: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.record(
            event_type="llm_call",
            problem_id=problem_id,
            component=component,
            assumption=assumption,
            why_selected=why_selected,
            expected_effect=expected_effect,
            observed_effect=observed_effect,
            status=status,
            verifier="runtime_trace",
            artifacts={"prompt_kind": prompt_kind, **(artifacts or {})},
            metadata=metadata or {},
        )

    def record_retrieval(
        self,
        *,
        problem_id: str,
        component: str,
        assumption: str,
        expected_effect: str,
        activated_assumption_ids: list[str],
        why_selected: str = "Graph retrieval is used to select assumption context for this problem.",
        observed_effect: str | None = None,
        artifacts: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.record(
            event_type="retrieval",
            problem_id=problem_id,
            component=component,
            assumption=assumption,
            why_selected=why_selected,
            expected_effect=expected_effect,
            observed_effect=observed_effect or f"activated={len(activated_assumption_ids)}",
            verifier="runtime_trace",
            artifacts={
                "activated_assumption_ids": activated_assumption_ids,
                **(artifacts or {}),
            },
            metadata=metadata or {},
        )

    def flush(self) -> dict:
        if not self.enabled:
            return {
                "eval_id": self.eval_id,
                "enabled": False,
                "event_count": 0,
                "writeback": False,
            }
        if self.events_out:
            self.events_out.parent.mkdir(parents=True, exist_ok=True)
            self.events_out.write_text(
                "\n".join(json.dumps(event, ensure_ascii=False, sort_keys=True) for event in self.events) + ("\n" if self.events else ""),
                encoding="utf-8",
            )
        store = JsonlGraphStore(self.graph_dir) if self.graph_dir else None
        payload = build_component_manifest_payload(
            eval_id=self.eval_id,
            events=self.events,
            store=store,
            writeback=self.writeback,
        )
        payload["enabled"] = True
        payload["events_out"] = str(self.events_out) if self.events_out else None
        payload["summary_out"] = str(self.summary_out) if self.summary_out else None
        if self.summary_out:
            self.summary_out.parent.mkdir(parents=True, exist_ok=True)
            self.summary_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload
