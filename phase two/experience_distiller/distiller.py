"""
Phase 2.2: Experience Distiller (heuristic MVP).

Given kept ExecutionRecords + evaluator results, produces UpdateCandidates
for the knowledge base.

MVP strategy (no LLM calls required):
  - For each contrast group (same problem, split outcomes across dispatchers):
      * Successful strategies: propose favorable conditions derived from
        problem's structural features.
      * Failing strategies: propose unfavorable conditions.
  - For unanimous-failure groups: attach conditions hinting at KB gaps
    to whichever strategy was selected.
  - Candidates are clustered by (target_strategy, placement, feature_signature)
    and merged — multiple supporting executions raise evidence strength.

Condition text is template-generated from feature thresholds. This is
good enough for a first KB evolution cycle; Phase 2.x can swap in LLM-based
condition phrasing without changing the integrator interface.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Each rule: (feature_key, high_threshold, low_threshold, text_when_high, text_when_low)
# The placement (favorable/unfavorable) is attached as a prefix at candidate build time.
FEATURE_RULES = [
    ("coupling_estimate", 0.65, 0.35,
     "组件之间存在较强耦合（coupling≥0.65）",
     "组件之间解耦清晰（coupling≤0.35）"),
    ("decomposability", 0.65, 0.35,
     "问题可清晰分解为子问题（decomposability≥0.65）",
     "问题难以分解（decomposability≤0.35）"),
    ("randomness_level", 0.65, 0.35,
     "任务高随机性（randomness≥0.65）",
     "任务随机性低（randomness≤0.35）"),
    ("information_completeness", 0.65, 0.35,
     "信息相对完整（info_completeness≥0.65）",
     "信息严重缺失（info_completeness≤0.35）"),
    ("reversibility", 0.65, 0.35,
     "操作可逆（reversibility≥0.65）",
     "操作不可逆（reversibility≤0.35）"),
]


@dataclass
class CandidateEvidence:
    execution_id: str
    problem_id: str
    outcome: str  # "success" | "failure"


@dataclass
class UpdateCandidate:
    candidate_id: str
    target_strategy: str
    placement: str          # "favorable" | "unfavorable"
    feature_key: str        # e.g. "coupling_estimate"
    feature_direction: str  # "high" | "low"
    condition_text: str
    supporting_evidence: List[CandidateEvidence] = field(default_factory=list)
    contradicting_evidence: List[CandidateEvidence] = field(default_factory=list)
    confidence: float = 0.0
    evidence_strength: str = "weak"   # weak | medium | strong
    stability_tier: str = "tentative"
    proposed_at: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


def _outcome_str(rec: Dict) -> str:
    return "success" if rec["outcome"]["success"] else "failure"


def _feature_direction(value: float, high: float, low: float) -> Optional[str]:
    if value >= high:
        return "high"
    if value <= low:
        return "low"
    return None


def _condition_text(rule: Tuple, direction: str) -> str:
    _, _, _, text_high, text_low = rule
    return text_high if direction == "high" else text_low


def _derive_candidates_from_record(rec: Dict) -> List[UpdateCandidate]:
    """Propose candidates from a single record based on outcome + features."""
    strategy = rec["strategy_selection"]["selected_strategy"]
    if strategy.startswith("SPECIAL") or strategy.startswith("COMP"):
        return []
    success = rec["outcome"]["success"]
    placement = "favorable" if success else "unfavorable"
    features = rec["task"].get("complexity_features", {})

    out: List[UpdateCandidate] = []
    for rule in FEATURE_RULES:
        key, high, low = rule[0], rule[1], rule[2]
        if key not in features:
            continue
        direction = _feature_direction(float(features[key]), high, low)
        if direction is None:
            continue

        text = _condition_text(rule, direction)
        cand = UpdateCandidate(
            candidate_id="",  # assigned at aggregation
            target_strategy=strategy,
            placement=placement,
            feature_key=key,
            feature_direction=direction,
            condition_text=text,
        )
        cand.supporting_evidence.append(CandidateEvidence(
            execution_id=rec["execution_id"],
            problem_id=rec["task"]["task_id"],
            outcome=_outcome_str(rec),
        ))
        out.append(cand)
    return out


def _merge_key(c: UpdateCandidate) -> Tuple:
    return (c.target_strategy, c.placement, c.feature_key, c.feature_direction)


def _evidence_strength(n_support: int, n_contradict: int) -> str:
    net = n_support - n_contradict
    if net >= 10:
        return "strong"
    if net >= 4:
        return "medium"
    return "weak"


def _confidence(n_support: int, n_contradict: int) -> float:
    # Beta posterior mean with a weakly informative prior (1,1).
    # This naturally caps unfavorable conditions with many contradictions:
    # e.g. 3 support / 10 contradict → confidence 0.27 (weak warning).
    # Favorable conditions which cleared the ratio gate will land 0.6-0.8.
    return (n_support + 1) / (n_support + n_contradict + 2)


def distill(
    kept_records: List[Dict],
    all_records_by_id: Dict[str, Dict],
) -> List[UpdateCandidate]:
    """
    kept_records: records marked keep=True by the evaluator
    all_records_by_id: full record map (for contradicting-evidence lookup)
    """
    # Step 1: propose raw candidates from kept records
    raw: List[UpdateCandidate] = []
    for rec in kept_records:
        raw.extend(_derive_candidates_from_record(rec))

    # Step 2: merge by (strategy, placement, feature, direction)
    merged: Dict[Tuple, UpdateCandidate] = {}
    for c in raw:
        k = _merge_key(c)
        if k not in merged:
            merged[k] = UpdateCandidate(
                candidate_id="",
                target_strategy=c.target_strategy,
                placement=c.placement,
                feature_key=c.feature_key,
                feature_direction=c.feature_direction,
                condition_text=c.condition_text,
            )
        merged[k].supporting_evidence.extend(c.supporting_evidence)

    # Step 3: collect contradicting evidence
    # A favorable claim "high coupling -> success" is contradicted by any
    # record with same (strategy, feature, direction) where outcome is failure.
    # A unfavorable claim is contradicted by success.
    for k, cand in merged.items():
        strategy, placement, fkey, fdir = k
        want_success = (placement == "favorable")

        # Search all records for counterexamples
        for rec in all_records_by_id.values():
            if rec["strategy_selection"]["selected_strategy"] != strategy:
                continue
            features = rec["task"].get("complexity_features", {})
            if fkey not in features:
                continue
            rule = next((r for r in FEATURE_RULES if r[0] == fkey), None)
            if rule is None:
                continue
            direction = _feature_direction(float(features[fkey]), rule[1], rule[2])
            if direction != fdir:
                continue

            outcome_match = rec["outcome"]["success"] == want_success
            if outcome_match:
                continue  # already counted as support (if kept) or neutral

            # Don't double-count: only add if not already in supporting
            ev_ids = {e.execution_id for e in cand.supporting_evidence}
            if rec["execution_id"] in ev_ids:
                continue

            cand.contradicting_evidence.append(CandidateEvidence(
                execution_id=rec["execution_id"],
                problem_id=rec["task"]["task_id"],
                outcome=_outcome_str(rec),
            ))

    # Step 4: finalize metadata
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    finals: List[UpdateCandidate] = []
    for k, cand in merged.items():
        n_s = len(cand.supporting_evidence)
        n_c = len(cand.contradicting_evidence)
        if n_s < 2:   # hard floor from config
            continue
        cand.confidence = round(_confidence(n_s, n_c), 3)
        cand.evidence_strength = _evidence_strength(n_s, n_c)
        cand.stability_tier = "tentative"
        cand.candidate_id = f"upd_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        cand.proposed_at = now
        finals.append(cand)

    return finals


def save_candidates(cands: List[UpdateCandidate], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cands:
        p = out_dir / f"{c.candidate_id}.json"
        p.write_text(json.dumps(c.to_dict(), ensure_ascii=False, indent=2))
    return len(cands)
