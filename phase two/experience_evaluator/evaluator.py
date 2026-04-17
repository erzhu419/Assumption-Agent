"""
Phase 2.1: Experience Evaluator.

Scores each ExecutionRecord for information value. High-info records pass
through to the distiller; low-info records are dropped.

Since the generator wrote records WITHOUT attribution, this evaluator works
in "contrast mode": it groups records by problem_id and uses the spread of
outcomes across dispatchers (DQN/SAC/RE-SAC) as the signal.

Information value rules (contrast mode):
  - Unanimous success   → 0.1  (confirming, low info)
  - Unanimous failure   → 0.5  (may hint at missing KB condition)
  - Split (success+failure on same problem) → 0.8  (strong contrast signal)
  - Partial / non-strategy actions → adjusted accordingly

Diminishing returns: when many problems share a similar feature profile and
the same outcome split, later records in that group are down-weighted.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class EvalResult:
    execution_id: str
    info_score: float
    tags: List[str] = field(default_factory=list)
    keep: bool = False
    reason: str = ""
    contrast_group: Optional[str] = None  # problem_id
    outcome_split: Optional[Tuple[int, int]] = None  # (success, failure) count in group


def _load_records(executions_dir: Path) -> List[Dict]:
    recs = []
    for f in sorted(executions_dir.glob("exec_*.json")):
        try:
            recs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return recs


def _group_by_problem(records: List[Dict]) -> Dict[str, List[Dict]]:
    grp: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        grp[r["task"]["task_id"]].append(r)
    return grp


def evaluate_all(
    executions_dir: Path,
    info_threshold: float = 0.3,
) -> List[EvalResult]:
    """Score every execution record by contrast within its problem group."""
    records = _load_records(executions_dir)
    groups = _group_by_problem(records)

    # Count similar feature profiles → diminishing returns
    # Bucket by (domain, difficulty, outcome_signature)
    profile_counts: Dict[Tuple, int] = defaultdict(int)

    results: List[EvalResult] = []

    for problem_id, recs in groups.items():
        successes = [r for r in recs if r["outcome"]["success"]]
        failures = [r for r in recs if not r["outcome"]["success"]]
        n_s, n_f = len(successes), len(failures)
        signature = (n_s, n_f)

        # Domain/difficulty key for diminishing returns
        sample = recs[0]
        domain = sample["task"].get("domain", "unknown")
        difficulty = sample["task"].get("difficulty", "medium")
        profile_key = (domain, difficulty, signature)
        profile_counts[profile_key] += 1
        # Diminishing factor: softer curve for failures (we want to keep
        # diverse failure evidence), harsher for unanimous success (confirmatory).
        raw_dim = 1.0 / (1.0 + math.log(1 + profile_counts[profile_key] - 1))
        is_failure_group = (n_f > 0)
        dim_factor = math.sqrt(raw_dim) if is_failure_group else raw_dim

        for r in recs:
            score, tags, reason = _score_one(r, n_s, n_f)
            score *= dim_factor

            keep = score >= info_threshold
            results.append(EvalResult(
                execution_id=r["execution_id"],
                info_score=round(score, 4),
                tags=tags,
                keep=keep,
                reason=reason,
                contrast_group=problem_id,
                outcome_split=(n_s, n_f),
            ))

    return results


def _score_one(r: Dict, n_success: int, n_failure: int) -> Tuple[float, List[str], str]:
    outcome = r["outcome"]
    meta = r.get("metadata", {})
    strategy = r["strategy_selection"]["selected_strategy"]
    score = 0.0
    tags: List[str] = []

    # Gate: SPECIAL_GATHER_INFO and compositions are meta-actions, not direct evidence
    if strategy.startswith("SPECIAL") or strategy.startswith("COMP"):
        tags.append("meta_action")
        return 0.05, tags, "meta action (low evidence value)"

    # Contrast signal: group has both success and failure on same problem
    if n_success > 0 and n_failure > 0:
        score += 0.8
        tags.append("contrast_group")
        reason = "split outcome in problem group (high contrast info)"
    elif n_failure > 0 and n_success == 0:
        # All dispatchers failed → strong signal of KB gap.
        # Failures are the real information source: we boost them,
        # and the distiller will build unfavorable conditions from them.
        score += 0.95
        tags.append("unanimous_failure")
        reason = "all dispatchers failed (KB gap candidate)"
    else:
        # All succeeded → low info, only useful for confidence reinforcement
        score += 0.1
        tags.append("unanimous_success")
        reason = "all dispatchers succeeded (confirmatory)"

    # Partial success softens the signal slightly
    if meta.get("partial_success"):
        score += 0.05
        tags.append("partial_success")

    # Consistency score: low consistency (annotators disagreed) => genuinely hard problem
    cs = outcome.get("consistency_score", 0.5)
    if cs < 0.3:
        score += 0.1
        tags.append("low_consistency")

    return score, tags, reason


def summarize(results: List[EvalResult]) -> Dict:
    kept = [r for r in results if r.keep]
    dropped = [r for r in results if not r.keep]
    tag_counts: Dict[str, int] = defaultdict(int)
    for r in results:
        for t in r.tags:
            tag_counts[t] += 1
    return {
        "total": len(results),
        "kept": len(kept),
        "dropped": len(dropped),
        "keep_rate": len(kept) / max(len(results), 1),
        "tag_counts": dict(tag_counts),
        "score_mean": sum(r.info_score for r in results) / max(len(results), 1),
    }
