"""
Phase 2.3: Knowledge Integrator.

Takes UpdateCandidates from pending_review and decides:
  - APPLY: write the new condition into kb/strategies/S*.json, log to change_history
  - REJECT: move to rejected/ with reason
  - PENDING_HUMAN: move to pending_human/ for review

Gates (in order):
  1. Evidence floor: n_support >= tier.min_evidence_to_modify
  2. Support/contradict ratio >= 2.0
  3. Conflict check: if a foundational condition on the same placement for the
     same strategy already contradicts the proposed condition, require human
     review rather than auto-apply.
  4. Duplicate check: if the KB already has a condition with the same
     feature_key/direction, merge (update supporting_cases, bump confidence
     by max_confidence_delta_per_update) rather than insert.

All applied changes append a JSONL line to change_history/<strategy_id>.jsonl.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class IntegrationDecision:
    candidate_id: str
    target_strategy: str
    action: str           # "apply_new" | "apply_merge" | "reject" | "human_review"
    reason: str
    change_id: Optional[str] = None


def _load_strategy(kb_dir: Path, strategy_id: str) -> Tuple[Optional[dict], Optional[Path]]:
    for f in kb_dir.glob(f"{strategy_id}_*.json"):
        return json.loads(f.read_text(encoding="utf-8")), f
    return None, None


def _similar_condition(
    existing: List[dict],
    feature_key: str,
    direction: str,
) -> Optional[dict]:
    """Find an existing condition tagged with same feature_key+direction."""
    for c in existing:
        meta = c.get("derived_from", {}) or {}
        if meta.get("feature_key") == feature_key and meta.get("feature_direction") == direction:
            return c
    return None


def _conflict_on_opposite_placement(
    strategy: dict,
    feature_key: str,
    direction: str,
    placement: str,
) -> Optional[dict]:
    """Check if the opposite placement already has a locked/foundational rule
    with the same feature_key+direction — that would be a direct conflict."""
    opp = "unfavorable" if placement == "favorable" else "favorable"
    opp_list = strategy.get("applicability_conditions", {}).get(opp, [])
    for c in opp_list:
        meta = c.get("derived_from", {}) or {}
        if meta.get("feature_key") != feature_key or meta.get("feature_direction") != direction:
            continue
        if c.get("locked") or c.get("stability_tier") == "foundational":
            return c
    return None


def _append_change_history(
    change_dir: Path,
    strategy_id: str,
    change_entry: dict,
):
    change_dir.mkdir(parents=True, exist_ok=True)
    path = change_dir / f"{strategy_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(change_entry, ensure_ascii=False) + "\n")


def integrate_candidate(
    cand: dict,
    kb_dir: Path,
    change_dir: Path,
    tier_config: dict,
) -> IntegrationDecision:
    strategy_id = cand["target_strategy"]
    placement = cand["placement"]
    feature_key = cand["feature_key"]
    direction = cand["feature_direction"]
    n_support = len(cand.get("supporting_evidence", []))
    n_contradict = len(cand.get("contradicting_evidence", []))

    strategy, path = _load_strategy(kb_dir, strategy_id)
    if strategy is None:
        return IntegrationDecision(cand["candidate_id"], strategy_id, "reject",
                                   f"strategy {strategy_id} not found in KB")

    tier_name = cand.get("stability_tier", "tentative")
    tier = tier_config.get(tier_name, tier_config["tentative"])

    # Gate 1: evidence floor
    if n_support < tier["min_evidence_to_modify"]:
        return IntegrationDecision(cand["candidate_id"], strategy_id, "reject",
                                   f"insufficient evidence ({n_support} < {tier['min_evidence_to_modify']})")

    # Gate 2: support/contradict ratio >= 2 for both placements.
    # (Phase 2 v3 experiment with relaxed unfavorable gate (0.3) showed this
    # hurts OOD — more warnings over-triggered. Keeping v2 settings.)
    ratio = n_support / max(n_contradict, 1)
    if ratio < 2.0:
        return IntegrationDecision(cand["candidate_id"], strategy_id, "reject",
                                   f"support/contradict ratio too low ({n_support}:{n_contradict})")

    # Gate 3: conflict with foundational rule on opposite placement
    conflict = _conflict_on_opposite_placement(strategy, feature_key, direction, placement)
    if conflict is not None:
        return IntegrationDecision(cand["candidate_id"], strategy_id, "human_review",
                                   f"conflicts with foundational condition {conflict.get('condition_id')}")

    # Gate 4: dedupe vs existing same-direction condition on same placement
    same_placement = strategy.setdefault("applicability_conditions", {}).setdefault(placement, [])
    existing = _similar_condition(same_placement, feature_key, direction)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    change_id = f"chg_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"

    if existing is not None:
        # Merge: bump confidence, extend supporting_cases
        cap = float(tier["max_confidence_delta_per_update"])
        old_conf = float(existing.get("confidence", 0.5))
        bump = min(cap, (float(cand["confidence"]) - old_conf))
        new_conf = max(0.0, min(1.0, old_conf + max(bump, 0.0)))
        existing["confidence"] = round(new_conf, 3)
        existing.setdefault("supporting_cases", []).extend(
            e["execution_id"] for e in cand.get("supporting_evidence", [])[:5]
        )
        existing["last_updated"] = now[:10]
        existing["version"] = int(existing.get("version", 1)) + 1

        strategy.setdefault("version", 1)
        strategy["version"] += 1
        path.write_text(json.dumps(strategy, ensure_ascii=False, indent=2))

        _append_change_history(change_dir, strategy_id, {
            "change_id": change_id,
            "timestamp": now,
            "type": "confidence_adjusted",
            "author": "phase2_auto",
            "candidate_id": cand["candidate_id"],
            "changes": {
                "condition_id": existing["condition_id"],
                "old_confidence": old_conf,
                "new_confidence": existing["confidence"],
            },
            "evidence_refs": [e["execution_id"] for e in cand.get("supporting_evidence", [])],
            "previous_version": strategy["version"] - 1,
            "new_version": strategy["version"],
        })
        return IntegrationDecision(cand["candidate_id"], strategy_id, "apply_merge",
                                   f"bumped confidence of {existing['condition_id']} to {existing['confidence']}",
                                   change_id=change_id)

    # Fresh condition: append
    new_condition_id = f"{strategy_id}_{placement[0].upper()}_AUTO_{uuid.uuid4().hex[:6]}"
    new_cond = {
        "condition_id": new_condition_id,
        "condition": cand["condition_text"],
        "source": "experience",
        "source_ref": f"candidate:{cand['candidate_id']}",
        "confidence": float(cand["confidence"]),
        "supporting_cases": [e["execution_id"] for e in cand.get("supporting_evidence", [])],
        "contradicting_cases": [e["execution_id"] for e in cand.get("contradicting_evidence", [])],
        "last_updated": now[:10],
        "version": 1,
        "status": "active",
        "locked": False,
        "stability_tier": tier_name,
        "derived_from": {
            "feature_key": feature_key,
            "feature_direction": direction,
        },
    }
    same_placement.append(new_cond)

    strategy.setdefault("version", 1)
    strategy["version"] += 1
    path.write_text(json.dumps(strategy, ensure_ascii=False, indent=2))

    _append_change_history(change_dir, strategy_id, {
        "change_id": change_id,
        "timestamp": now,
        "type": "condition_added",
        "author": "phase2_auto",
        "candidate_id": cand["candidate_id"],
        "changes": {
            "added": {
                "condition_id": new_condition_id,
                "condition": cand["condition_text"],
                "placement": placement,
                "confidence": cand["confidence"],
                "stability_tier": tier_name,
            }
        },
        "evidence_refs": [e["execution_id"] for e in cand.get("supporting_evidence", [])],
        "previous_version": strategy["version"] - 1,
        "new_version": strategy["version"],
    })
    return IntegrationDecision(cand["candidate_id"], strategy_id, "apply_new",
                               f"added {new_condition_id} to {placement}",
                               change_id=change_id)


def route_candidate_file(
    cand_path: Path,
    decision: IntegrationDecision,
    applied_dir: Path,
    rejected_dir: Path,
    human_dir: Path,
):
    target = {
        "apply_new": applied_dir,
        "apply_merge": applied_dir,
        "reject": rejected_dir,
        "human_review": human_dir,
    }[decision.action]
    target.mkdir(parents=True, exist_ok=True)
    shutil.move(str(cand_path), str(target / cand_path.name))
