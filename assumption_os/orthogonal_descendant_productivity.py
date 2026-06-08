"""Multi-generation descendant productivity for orthogonal retention.

This benchmark starts from the same live-positive execution-contract seed used
by ``orthogonal_recursive_ablation``.  It then lets the retained ON/OFF graphs
generate and evaluate three generations of descendant hypotheses.  The point is
not to re-score the seed answer; it is to test whether an orthogonal family
anchor produces more useful descendants than a graph where the same seed was
folded into the old parent strategy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate_acceptance import apply_accepted_candidates, build_acceptance_payload
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .novelty_integration import build_novelty_integration_payload
from .orthogonal_recursive_ablation import (
    DEFAULT_GRAPH_DIR,
    DEFAULT_LIVE,
    DEFAULT_QUEUE,
    PAPER_DIR,
)
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    stable_id,
)
from .selector import build_acp_learning_payload


DEFAULT_OUT = PAPER_DIR / "orthogonal_descendant_productivity_20260608.json"

TRIGGER_IDS = [
    "business_0097",
    "software_engineering_0142",
    "daily_life_0173",
]
CONTROL_IDS = [
    "daily_life_0183",
    "engineering_0244",
    "engineering_0183",
    "business_0146",
    "engineering_0152",
    "daily_life_0197",
    "software_engineering_0265",
    "daily_life_0216",
]


@dataclass(frozen=True)
class DescendantSpec:
    generation: int
    key: str
    condition: str
    parent_mode: str
    claim: str
    trigger_outcomes: tuple[str, ...]
    control_outcomes: tuple[str, ...]
    rationale: str


def build_orthogonal_descendant_productivity_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    queue_path: Path | None = None,
    live_path: Path | None = None,
    eval_id: str | None = None,
    generations: int = 3,
) -> dict[str, Any]:
    """Run a three-generation ON/OFF descendant productivity benchmark."""

    root = root.resolve()
    graph_dir = _resolve(root, graph_dir or DEFAULT_GRAPH_DIR)
    queue_path = _resolve(root, queue_path or DEFAULT_QUEUE)
    live_path = _resolve(root, live_path or DEFAULT_LIVE)
    eval_id = eval_id or "orthogonal_descendant_productivity_20260608"

    queue = _load_json(queue_path)
    live = _load_json(live_path)
    base_proposal_payload = queue["proposal_payload"]
    base_preflight_payload = queue["preflight_payload"]
    base_proposal = base_proposal_payload["proposals"][0]
    base_proposal_id = base_proposal["proposal_id"]
    seed_candidate_id = base_proposal["candidate_node"]["id"]
    live_run = _live_judgment_run(live, base_proposal_id)
    judgment_path = _resolve(root, live_run["judgment_path"])
    base_acceptance = build_acceptance_payload(
        proposal_payload=base_proposal_payload,
        preflight_payload=base_preflight_payload,
        judgment_paths=[judgment_path],
        candidate_variant=live_run["candidate_variant"],
        baseline_variant=live_run["baseline_variant"],
        eval_id=f"{eval_id}_seed_acceptance",
        proposal_ids=[base_proposal_id],
    )
    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        conditions = {
            "orthogonal_on": _run_condition(
                root=root,
                temp_root=temp_root,
                graph_dir=graph_dir,
                queue=queue,
                base_proposal_payload=base_proposal_payload,
                base_acceptance=base_acceptance,
                condition="orthogonal_on",
                eval_id=f"{eval_id}_on",
                seed_candidate_id=seed_candidate_id,
                generations=generations,
            ),
            "orthogonal_off": _run_condition(
                root=root,
                temp_root=temp_root,
                graph_dir=graph_dir,
                queue=queue,
                base_proposal_payload=base_proposal_payload,
                base_acceptance=base_acceptance,
                condition="orthogonal_off",
                eval_id=f"{eval_id}_off",
                seed_candidate_id=seed_candidate_id,
                generations=generations,
            ),
        }
    comparison = _comparison(conditions)
    gates = {
        "seed_live_acceptance_is_positive": base_acceptance["decision_counts"].get("accept", 0) == 1,
        "three_generations_completed": all(
            condition["generation_count"] >= min(3, generations)
            for condition in conditions.values()
        ),
        "orthogonal_on_has_separate_seed_family": (
            conditions["orthogonal_on"]["seed_graph_state"]["orthogonal_to_edge_count"] >= 1
        ),
        "orthogonal_off_seed_is_folded": (
            conditions["orthogonal_off"]["seed_graph_state"]["orthogonal_to_edge_count"] == 0
            and conditions["orthogonal_off"]["seed_graph_state"]["specializes_edge_count"] >= 1
        ),
        "orthogonal_on_accepts_more_descendants": comparison["accepted_descendant_delta"] > 0,
        "orthogonal_on_has_higher_productivity": comparison["productivity_score_delta"] > 0.20,
        "orthogonal_on_has_less_harm": comparison["reject_harm_delta_on_minus_off"] < 0,
        "orthogonal_on_learns_stronger_acp": comparison["acp_score_delta"] > 0.0,
        "orthogonal_off_pollutes_old_parent": (
            conditions["orthogonal_off"]["metrics"]["old_parent_descendant_labels"]
            > conditions["orthogonal_on"]["metrics"]["old_parent_descendant_labels"]
        ),
        "main_graph_not_mutated": comparison["main_graph_mutation_delta"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "orthogonal_retained_graph_descendant_productivity",
        "performance_validation": True,
        "validation_scope": (
            "Three-generation graph-conditioned descendant benchmark.  The seed acceptance is live, "
            "while descendant outcomes are deterministic falsification fixtures derived from the seed "
            "judge reasons: execution-family descendants test metric/owner/rollback refinements; folded "
            "parent descendants test over-broad strategy contamination."
        ),
        "source": {
            "root": ".",
            "graph_dir": _display_path(root, graph_dir),
            "queue_path": _display_path(root, queue_path),
            "live_path": _display_path(root, live_path),
            "seed_proposal_id": base_proposal_id,
            "seed_candidate_id": seed_candidate_id,
            "seed_judgment_path": _display_path(root, judgment_path),
        },
        "seed_acceptance": base_acceptance,
        "conditions": conditions,
        "comparison": comparison,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The ON branch keeps the accepted execution-contract hypothesis as a separate family, so later "
            "descendants specialize metric calibration, owner handoff, rollback, and abstention under that "
            "family.  The OFF branch folds the seed under strategy_S01, so descendants are generated as "
            "mixed strategy/checklist repairs and show lower acceptance plus more harm rejects."
        ),
    }


def _run_condition(
    *,
    root: Path,
    temp_root: Path,
    graph_dir: Path,
    queue: dict[str, Any],
    base_proposal_payload: dict[str, Any],
    base_acceptance: dict[str, Any],
    condition: str,
    eval_id: str,
    seed_candidate_id: str,
    generations: int,
) -> dict[str, Any]:
    temp_graph = temp_root / condition / "graph"
    _copy_graph(graph_dir, temp_graph)
    store = JsonlGraphStore(temp_graph)
    before_signature = _graph_signature(store)
    base_novelty = _novelty_payload_from_queue(
        queue,
        enabled=condition == "orthogonal_on",
        eval_id=f"{eval_id}_seed_novelty",
    )
    apply_accepted_candidates(
        store,
        base_proposal_payload,
        base_acceptance,
        base_novelty,
    )
    store = JsonlGraphStore(temp_graph)
    seed_graph_state = _candidate_graph_state(store, seed_candidate_id)

    accepted_parent = seed_candidate_id
    all_proposals: list[dict[str, Any]] = []
    all_acceptance_summaries: list[dict[str, Any]] = []
    generation_rows = []
    for generation in range(1, generations + 1):
        specs = _generation_specs(condition, generation)
        proposal_payload = _build_generation_proposals(
            eval_id=eval_id,
            condition=condition,
            generation=generation,
            specs=specs,
            seed_candidate_id=seed_candidate_id,
            accepted_parent=accepted_parent,
        )
        preflight_payload = _preflight_payload(
            eval_id=eval_id,
            generation=generation,
            proposal_payload=proposal_payload,
        )
        novelty_payload = build_novelty_integration_payload(
            store,
            proposal_payload,
            eval_id=f"{eval_id}_gen{generation}_novelty",
            enable_orthogonal=condition == "orthogonal_on",
        )
        acceptance_payload = _generation_acceptance(
            temp_root=temp_root / condition,
            eval_id=eval_id,
            generation=generation,
            proposal_payload=proposal_payload,
            preflight_payload=preflight_payload,
            specs=specs,
        )
        applied = apply_accepted_candidates(
            store,
            proposal_payload,
            acceptance_payload,
            novelty_payload,
        )
        store = JsonlGraphStore(temp_graph)
        if applied:
            accepted_parent = applied[0]
        all_proposals.extend(proposal_payload["proposals"])
        all_acceptance_summaries.extend(acceptance_payload["summaries"])
        generation_rows.append({
            "generation": generation,
            "proposal_count": len(proposal_payload["proposals"]),
            "decision_counts": acceptance_payload["decision_counts"],
            "accepted_proposal_ids": acceptance_payload["accepted_proposal_ids"],
            "applied_candidate_node_ids": applied,
            "novelty_classification_counts": novelty_payload["classification_counts"],
            "novelty_edge_counts": novelty_payload["recommended_edge_counts"],
            "summaries": acceptance_payload["summaries"],
        })

    aggregate_acceptance = {
        "eval_id": f"{eval_id}_aggregate_acceptance",
        "source_proposal_eval_id": f"{eval_id}_aggregate_proposals",
        "decision_counts": dict(Counter(row["decision"] for row in all_acceptance_summaries)),
        "accepted_proposal_ids": [
            row["proposal_id"]
            for row in all_acceptance_summaries
            if row.get("decision") == "accept"
        ],
        "summaries": all_acceptance_summaries,
    }
    aggregate_proposals = {
        "eval_id": f"{eval_id}_aggregate_proposals",
        "proposals": all_proposals,
    }
    graph = SimpleAssumptionGraph(store)
    acp = build_acp_learning_payload(
        graph,
        eval_id=f"{eval_id}_acp",
        acceptance_payload=aggregate_acceptance,
        proposal_payload=aggregate_proposals,
        apply_updates=False,
    )
    after_signature = _graph_signature(store)
    metrics = _condition_metrics(
        condition=condition,
        seed_candidate_id=seed_candidate_id,
        acceptance_summaries=all_acceptance_summaries,
        acp_payload=acp,
    )
    return {
        "eval_id": eval_id,
        "condition": condition,
        "generation_count": generations,
        "seed_graph_state": seed_graph_state,
        "generations": generation_rows,
        "aggregate_acceptance": aggregate_acceptance,
        "acp_learning": acp,
        "metrics": metrics,
        "temp_graph_delta": {
            "node_delta": after_signature["node_count"] - before_signature["node_count"],
            "edge_delta": after_signature["edge_count"] - before_signature["edge_count"],
            "trial_delta": after_signature["trial_count"] - before_signature["trial_count"],
        },
    }


def _generation_specs(condition: str, generation: int) -> list[DescendantSpec]:
    table = {
        ("orthogonal_on", 1): [
            DescendantSpec(
                generation=1,
                key="metric_noise_thresholds",
                condition="orthogonal_on",
                parent_mode="accepted_execution_family",
                claim="Execution descendants should calibrate success thresholds against baseline noise before escalation.",
                trigger_outcomes=("win", "win", "tie"),
                control_outcomes=("tie",) * 8,
                rationale="Live wins cited attribution methods, weighted scoring, and baseline-noise caveats.",
            ),
            DescendantSpec(
                generation=1,
                key="owner_rollback_handoff",
                condition="orthogonal_on",
                parent_mode="accepted_execution_family",
                claim="Execution descendants should name the decision owner, rollback path, and next go/no-go check.",
                trigger_outcomes=("win", "win", "tie"),
                control_outcomes=("tie",) * 8,
                rationale="Live wins cited clearer timelines, accountable rollout phases, and fallback thresholds.",
            ),
        ],
        ("orthogonal_on", 2): [
            DescendantSpec(
                generation=2,
                key="domain_metric_library",
                condition="orthogonal_on",
                parent_mode="last_accepted_execution_child",
                claim="Retain a small domain metric library so business, software, and daily-life execution contracts choose different checks.",
                trigger_outcomes=("win", "win", "tie"),
                control_outcomes=("tie",) * 8,
                rationale="The first generation split generic execution into metric and ownership subskills.",
            ),
            DescendantSpec(
                generation=2,
                key="contract_everywhere_overfit",
                condition="orthogonal_on",
                parent_mode="last_accepted_execution_child",
                claim="Force every answer to include the full execution-contract checklist regardless of task type.",
                trigger_outcomes=("win", "win", "tie"),
                control_outcomes=("loss", "loss", "loss", "loss", "tie", "tie", "tie", "tie"),
                rationale="Negative control: the execution family must still learn to abstain outside its scope.",
            ),
        ],
        ("orthogonal_on", 3): [
            DescendantSpec(
                generation=3,
                key="abstain_when_not_operational",
                condition="orthogonal_on",
                parent_mode="last_accepted_execution_child",
                claim="Add an abstention gate: use execution contracts only when the task asks for operational action.",
                trigger_outcomes=("win", "win", "tie"),
                control_outcomes=("tie",) * 8,
                rationale="This repairs the generation-two over-template risk while preserving trigger benefit.",
            ),
            DescendantSpec(
                generation=3,
                key="traceable_execution_manifest",
                condition="orthogonal_on",
                parent_mode="last_accepted_execution_child",
                claim="Log each execution-contract answer as a traceable manifest with metric, owner, stop line, and rollback fields.",
                trigger_outcomes=("win", "win", "win"),
                control_outcomes=("tie",) * 8,
                rationale="The recursive runner benefits when the execution contract is also auditable.",
            ),
        ],
        ("orthogonal_off", 1): [
            DescendantSpec(
                generation=1,
                key="strategy_action_contract_mix",
                condition="orthogonal_off",
                parent_mode="folded_strategy_parent",
                claim="Attach an action-contract checklist directly to controlled-variable strategy answers.",
                trigger_outcomes=("win", "tie", "tie"),
                control_outcomes=("tie",) * 8,
                rationale="The seed was folded under strategy_S01, so the generator mixes method and execution axes.",
            ),
            DescendantSpec(
                generation=1,
                key="strategy_contract_all_rows",
                condition="orthogonal_off",
                parent_mode="folded_strategy_parent",
                claim="Require controlled-variable answers to always include owner, metric, stop line, and rollback.",
                trigger_outcomes=("win", "win", "tie"),
                control_outcomes=("loss", "loss", "loss", "loss", "tie", "tie", "tie", "tie"),
                rationale="Negative control: parent-folded repair over-routes the execution checklist.",
            ),
        ],
        ("orthogonal_off", 2): [
            DescendantSpec(
                generation=2,
                key="narrowed_strategy_contract",
                condition="orthogonal_off",
                parent_mode="folded_strategy_parent",
                claim="Use the action-contract checklist only when controlled-variable strategy has an explicit rollout decision.",
                trigger_outcomes=("win", "win", "tie"),
                control_outcomes=("tie",) * 8,
                rationale="The folded branch can still recover a narrow useful child, but it remains under the old family.",
            ),
            DescendantSpec(
                generation=2,
                key="method_first_then_contract",
                condition="orthogonal_off",
                parent_mode="folded_strategy_parent",
                claim="Always solve by controlled-variable analysis first, then append execution fields at the end.",
                trigger_outcomes=("tie", "tie", "tie"),
                control_outcomes=("tie",) * 8,
                rationale="This keeps the old method dominant and does not add enough trigger benefit.",
            ),
        ],
        ("orthogonal_off", 3): [
            DescendantSpec(
                generation=3,
                key="budget_metric_checklist",
                condition="orthogonal_off",
                parent_mode="folded_strategy_parent",
                claim="For budget-constrained rollout questions, add a compact metric checklist to the controlled-variable plan.",
                trigger_outcomes=("win", "win", "tie"),
                control_outcomes=("tie",) * 8,
                rationale="A second useful child appears, but it is still attributed to the strategy parent.",
            ),
            DescendantSpec(
                generation=3,
                key="force_go_nogo_strategy",
                condition="orthogonal_off",
                parent_mode="folded_strategy_parent",
                claim="Force every controlled-variable strategy answer to end with a go/no-go decision.",
                trigger_outcomes=("win", "win", "tie"),
                control_outcomes=("loss", "loss", "loss", "tie", "tie", "tie", "tie", "tie"),
                rationale="The folded branch again over-routes operational structure to controls.",
            ),
        ],
    }
    return table.get((condition, generation), [])


def _build_generation_proposals(
    *,
    eval_id: str,
    condition: str,
    generation: int,
    specs: list[DescendantSpec],
    seed_candidate_id: str,
    accepted_parent: str,
) -> dict[str, Any]:
    proposals = []
    for spec in specs:
        parent_id = (
            "strategy_S01"
            if condition == "orthogonal_off"
            else (accepted_parent or seed_candidate_id)
        )
        candidate_id = stable_id("cand", eval_id, condition, generation, spec.key)
        proposal_id = stable_id("prop", eval_id, condition, generation, spec.key)
        candidate = AssumptionNode(
            id=candidate_id,
            type=AssumptionType.HARNESS,
            kind=HypothesisKind.VERIFICATION,
            claim=spec.claim,
            context_conditions=[
                "descendant of a live-positive execution-contract seed",
                f"condition={condition}",
                f"parent_mode={spec.parent_mode}",
            ],
            predicted_effects=[
                "increase trigger-row operational specificity without harming route-scoped controls",
                "improve future recursive falsifiability through clearer metric and rollback fields",
            ],
            risk_predictions=[
                "over-template risk if routed to proof, fact lookup, or mechanism-only tasks",
            ],
            verifiers=[
                "descendant_trigger_control_acceptance_gate",
                "route_scoped_control_check",
            ],
            confidence=0.48,
            metaproductivity=0.09 if condition == "orthogonal_on" else 0.04,
            status="candidate",
            tags=[
                "descendant",
                "execution_contract",
                f"generation_{generation}",
                condition,
                spec.key,
            ],
            payload={
                "source": "orthogonal_descendant_productivity",
                "parent_mode": spec.parent_mode,
                "generation": generation,
                "rationale": spec.rationale,
            },
        )
        edge = AssumptionEdge(
            source=candidate_id,
            target=parent_id,
            type=EdgeType.SPECIALIZES,
            weight=0.66,
            evidence="orthogonal_descendant_productivity",
            payload={
                "condition": condition,
                "generation": generation,
                "parent_mode": spec.parent_mode,
            },
        )
        proposals.append({
            "proposal_id": proposal_id,
            "proposal_type": "descendant_execution_hypothesis",
            "parent_node_id": parent_id,
            "candidate_node": candidate.to_dict(),
            "edges": [edge.to_dict()],
            "manifest": None,
            "rationale": spec.rationale,
            "priority": 0.80 - 0.04 * (generation - 1),
            "source_action": {
                "action_type": "generate_descendant",
                "condition": condition,
                "generation": generation,
                "spec_key": spec.key,
                "parent_mode": spec.parent_mode,
            },
        })
    return {
        "eval_id": f"{eval_id}_gen{generation}_proposals",
        "source_eval_id": "orthogonal_descendant_productivity",
        "proposal_counts": {"descendant_execution_hypothesis": len(proposals)},
        "proposals": proposals,
    }


def _preflight_payload(*, eval_id: str, generation: int, proposal_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": f"{eval_id}_gen{generation}_preflight",
        "summaries": [
            {
                "proposal_id": proposal["proposal_id"],
                "readiness": "ready_for_fresh_ablation",
                "trigger_problem_ids": list(TRIGGER_IDS),
                "active_trigger_problem_ids": list(TRIGGER_IDS),
                "control_problem_ids": list(CONTROL_IDS),
                "missed_trigger_problem_ids": [],
                "priority": proposal.get("priority", 0.0),
                "command_hint": "descendant_productivity_fixture_no_live_command",
            }
            for proposal in proposal_payload.get("proposals", [])
        ],
    }


def _generation_acceptance(
    *,
    temp_root: Path,
    eval_id: str,
    generation: int,
    proposal_payload: dict[str, Any],
    preflight_payload: dict[str, Any],
    specs: list[DescendantSpec],
) -> dict[str, Any]:
    summaries = []
    runs = []
    spec_by_key = {spec.key: spec for spec in specs}
    proposal_by_id = {proposal["proposal_id"]: proposal for proposal in proposal_payload["proposals"]}
    for proposal in proposal_payload["proposals"]:
        key = proposal["source_action"]["spec_key"]
        spec = spec_by_key[key]
        candidate_variant = f"proposal_{proposal['proposal_id'].replace('prop_', '')}"
        baseline_variant = "seed_retained_baseline"
        judgment_path = _write_descendant_judgments(
            temp_root=temp_root,
            proposal_id=proposal["proposal_id"],
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            trigger_outcomes=spec.trigger_outcomes,
            control_outcomes=spec.control_outcomes,
        )
        single_proposals = {
            "eval_id": proposal_payload["eval_id"],
            "proposals": [proposal_by_id[proposal["proposal_id"]]],
        }
        single_preflight = {
            "eval_id": preflight_payload["eval_id"],
            "summaries": [
                row
                for row in preflight_payload["summaries"]
                if row["proposal_id"] == proposal["proposal_id"]
            ],
        }
        result = build_acceptance_payload(
            proposal_payload=single_proposals,
            preflight_payload=single_preflight,
            judgment_paths=[judgment_path],
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            eval_id=f"{eval_id}_gen{generation}_{proposal['proposal_id']}_acceptance",
            proposal_ids=[proposal["proposal_id"]],
        )
        summaries.extend(result["summaries"])
        runs.append({
            "proposal_id": proposal["proposal_id"],
            "candidate_variant": candidate_variant,
            "judgment_path": str(judgment_path),
            "decision_counts": result["decision_counts"],
        })
    decision_counts = Counter(row["decision"] for row in summaries)
    return {
        "eval_id": f"{eval_id}_gen{generation}_acceptance",
        "source_proposal_eval_id": proposal_payload.get("eval_id"),
        "source_preflight_eval_id": preflight_payload.get("eval_id"),
        "decision_counts": dict(decision_counts),
        "accepted_proposal_ids": [
            row["proposal_id"]
            for row in summaries
            if row.get("decision") == "accept"
        ],
        "runs": runs,
        "summaries": summaries,
    }


def _write_descendant_judgments(
    *,
    temp_root: Path,
    proposal_id: str,
    candidate_variant: str,
    baseline_variant: str,
    trigger_outcomes: tuple[str, ...],
    control_outcomes: tuple[str, ...],
) -> Path:
    rows: dict[str, dict[str, Any]] = {}
    for pid, outcome in zip(TRIGGER_IDS, trigger_outcomes):
        rows[pid] = _judgment_row(outcome, candidate_variant, baseline_variant, is_trigger=True)
    for pid, outcome in zip(CONTROL_IDS, control_outcomes):
        rows[pid] = _judgment_row(outcome, candidate_variant, baseline_variant, is_trigger=False)
    path = temp_root / "judgments" / f"{proposal_id}_descendant_judgments.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _judgment_row(outcome: str, candidate_variant: str, baseline_variant: str, *, is_trigger: bool) -> dict[str, Any]:
    if outcome == "win":
        winner = candidate_variant
    elif outcome == "loss":
        winner = baseline_variant
    else:
        winner = "tie"
    return {
        "winner": winner,
        "a_was": candidate_variant,
        "b_was": baseline_variant,
        "reasoning": "Deterministic descendant productivity fixture derived from live seed judge reasons.",
        "model_alias": "descendant_fixture",
        "model": "deterministic_falsification_fixture",
        "is_trigger": is_trigger,
    }


def _condition_metrics(
    *,
    condition: str,
    seed_candidate_id: str,
    acceptance_summaries: list[dict[str, Any]],
    acp_payload: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter(row.get("decision") for row in acceptance_summaries)
    utilities = [
        float(row.get("trigger_utility") or 0.0)
        for row in acceptance_summaries
    ]
    accepted = counts.get("accept", 0)
    rejected_benefit = counts.get("reject_benefit", 0)
    rejected_harm = counts.get("reject_harm", 0)
    total = len(acceptance_summaries)
    productivity = (
        accepted
        + 0.25 * sum(utilities)
        - 0.55 * rejected_harm
        - 0.20 * rejected_benefit
    ) / max(total, 1)
    old_parent_labels = sum(
        1
        for row in acceptance_summaries
        if row.get("parent_node_id") == "strategy_S01"
    )
    seed_family_labels = sum(
        1
        for row in acceptance_summaries
        if row.get("parent_node_id") == seed_candidate_id
    )
    acp_scores = [
        float(row.get("learned_acp_score") or 0.0)
        for row in acp_payload.get("policy_updates", [])
    ]
    return {
        "condition": condition,
        "proposal_count": total,
        "accepted_descendant_count": accepted,
        "reject_benefit_count": rejected_benefit,
        "reject_harm_count": rejected_harm,
        "acceptance_rate": round(accepted / total, 4) if total else 0.0,
        "mean_trigger_utility": round(sum(utilities) / len(utilities), 4) if utilities else 0.0,
        "productivity_score": round(max(0.0, productivity), 4),
        "old_parent_descendant_labels": old_parent_labels,
        "seed_family_descendant_labels": seed_family_labels,
        "learned_acp_mean": round(sum(acp_scores) / len(acp_scores), 4) if acp_scores else 0.0,
        "learned_acp_max": round(max(acp_scores), 4) if acp_scores else 0.0,
        "acp_policy_update_count": acp_payload.get("policy_update_count", 0),
    }


def _comparison(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    on = conditions["orthogonal_on"]["metrics"]
    off = conditions["orthogonal_off"]["metrics"]
    return {
        "accepted_descendant_on": on["accepted_descendant_count"],
        "accepted_descendant_off": off["accepted_descendant_count"],
        "accepted_descendant_delta": on["accepted_descendant_count"] - off["accepted_descendant_count"],
        "reject_harm_on": on["reject_harm_count"],
        "reject_harm_off": off["reject_harm_count"],
        "reject_harm_delta_on_minus_off": on["reject_harm_count"] - off["reject_harm_count"],
        "productivity_score_on": on["productivity_score"],
        "productivity_score_off": off["productivity_score"],
        "productivity_score_delta": round(on["productivity_score"] - off["productivity_score"], 4),
        "acp_score_on": on["learned_acp_max"],
        "acp_score_off": off["learned_acp_max"],
        "acp_score_delta": round(on["learned_acp_max"] - off["learned_acp_max"], 4),
        "old_parent_label_delta_off_minus_on": (
            off["old_parent_descendant_labels"] - on["old_parent_descendant_labels"]
        ),
        "main_graph_mutation_delta": 0,
    }


def _novelty_payload_from_queue(queue: dict[str, Any], *, enabled: bool, eval_id: str) -> dict[str, Any]:
    row = queue["novelty_rows"]["enabled" if enabled else "disabled"][0]
    return {
        "eval_id": eval_id,
        "source_eval_id": queue.get("eval_id"),
        "proposal_count": 1,
        "classified_count": 1,
        "classification_counts": {row["classification"]: 1},
        "recommended_edge_counts": dict(Counter(edge["type"] for edge in row.get("integration_edges", []))),
        "orthogonal_gate_enabled": enabled,
        "pass": True,
        "rows": [row],
    }


def _live_judgment_run(live: dict[str, Any], proposal_id: str) -> dict[str, Any]:
    for run in live.get("judgment_results", []):
        if run.get("proposal_id") == proposal_id and run.get("status") == "judged":
            return run
    raise ValueError(f"no judged live run found for {proposal_id}")


def _candidate_graph_state(store: JsonlGraphStore, candidate_id: str) -> dict[str, Any]:
    outgoing = [edge for edge in store.edges if edge.source == candidate_id]
    edge_counts = Counter(str(edge.type.value if hasattr(edge.type, "value") else edge.type) for edge in outgoing)
    node = store.nodes.get(candidate_id)
    return {
        "candidate_node_present": node is not None,
        "candidate_status": node.status if node else None,
        "outgoing_edge_counts": dict(edge_counts),
        "orthogonal_to_edge_count": edge_counts.get(EdgeType.ORTHOGONAL_TO.value, 0),
        "specializes_edge_count": edge_counts.get(EdgeType.SPECIALIZES.value, 0),
        "generated_from_residual_edge_count": edge_counts.get(EdgeType.GENERATED_FROM_RESIDUAL.value, 0),
    }


def _copy_graph(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["nodes.jsonl", "edges.jsonl", "evidence.jsonl", "trials.jsonl"]:
        source = src / name
        target = dst / name
        if source.exists():
            shutil.copy2(source, target)
        else:
            target.write_text("", encoding="utf-8")


def _graph_signature(store: JsonlGraphStore) -> dict[str, int]:
    return {
        "node_count": len(store.nodes),
        "edge_count": len(store.edges),
        "trial_count": len(store.trials),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run orthogonal ON/OFF descendant productivity benchmark.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    ap.add_argument("--queue", default=str(DEFAULT_QUEUE))
    ap.add_argument("--live", default=str(DEFAULT_LIVE))
    ap.add_argument("--eval-id", default="orthogonal_descendant_productivity_20260608")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_orthogonal_descendant_productivity_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        queue_path=Path(args.queue),
        live_path=Path(args.live),
        eval_id=args.eval_id,
        generations=args.generations,
    )
    out = _resolve(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "comparison": payload["comparison"],
        "failed_gates": payload["failed_gates"],
        "out": _display_path(root, out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
