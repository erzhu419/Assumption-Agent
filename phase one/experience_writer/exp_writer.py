"""
Phase 1 -> Phase 2 bridge: write per-execution records.

Each record conforms to the Phase 0 ExecutionRecord schema
(see phase_zero_dev_doc.md section 2.7). The attribution field is
left sparse on write — Phase 2's distiller fills it via LLM analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExecutionRecord:
    execution_id: str
    timestamp: str
    task: Dict
    strategy_selection: Dict
    execution_trajectory: Dict
    outcome: Dict
    attribution: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


class ExperienceWriter:
    """Append-only writer for execution records.

    One JSON file per record (filename = execution_id) so each write
    is atomic and the distiller can pull files in shard-friendly batches.
    """

    def __init__(self, output_dir: Path, writer_version: str = "p1.v1"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer_version = writer_version
        self._seq = 0

    def _make_id(self, task_id: str) -> str:
        self._seq += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"exec_{ts}_{self._seq:06d}_{task_id}"

    def write(
        self,
        *,
        task: Dict,
        selected_strategy: str,
        selector_confidence: float,
        alternatives: List[str],
        outcome_success: bool,
        evaluation_score: float,
        consistency_score: float,
        failure_reason: str = "",
        wall_clock_seconds: float = 0.0,
        steps_taken: int = 1,
        extra_metadata: Optional[Dict] = None,
    ) -> Path:
        """Write one execution record and return its path."""
        task_id = task.get("problem_id", "unknown")
        exec_id = self._make_id(task_id)

        record = ExecutionRecord(
            execution_id=exec_id,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            task={
                "task_id": task_id,
                "description": task.get("description", ""),
                "domain": task.get("domain", "unknown"),
                "difficulty": task.get("difficulty", "medium"),
                "complexity_features": task.get("complexity_features", {}),
            },
            strategy_selection={
                "selected_strategy": selected_strategy,
                "selector_confidence": float(selector_confidence),
                "alternatives_considered": list(alternatives),
                "selection_reason": "dispatcher_policy",
            },
            execution_trajectory={
                "steps": [],
                "total_steps": int(steps_taken),
                "wall_clock_seconds": float(wall_clock_seconds),
            },
            outcome={
                "success": bool(outcome_success),
                "partial_success": bool((not outcome_success) and evaluation_score > 0),
                "evaluation_score": float(evaluation_score),
                "consistency_score": float(consistency_score),
                "failure_reason": failure_reason,
                "root_cause_type": None,
            },
            attribution={
                "matched_conditions": [],
                "violated_conditions": [],
                "newly_discovered_condition_candidates": [],
                "filled_by": None,
            },
            metadata={
                "writer_version": self.writer_version,
                "human_reviewed": False,
                "distilled_into_kb": False,
                "distillation_ref": None,
                **(extra_metadata or {}),
            },
        )

        path = self.output_dir / f"{exec_id}.json"
        path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2))
        return path

    def close(self):
        pass
