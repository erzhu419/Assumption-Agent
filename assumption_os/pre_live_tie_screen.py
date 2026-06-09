"""Pre-live screen for tie-prone or underpowered descendant proposals.

The recursive runner can already propose descendants and send them to live
ablation.  This module adds a cheap budget gate before that step: use prior
accepted/rejected sibling evidence to decide whether a new descendant should
run live now, be expanded first, or be blocked as a likely low-benefit repeat.

The default payload is a compact replay of the orthogonal descendant live line.
It intentionally keeps only outcome counts and artifact paths, not API data or
raw model answers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "pre_live_tie_screen_20260609.json"


@dataclass(frozen=True)
class PreLiveReplayCase:
    order: int
    proposal_id: str
    candidate_node_id: str
    parent_node_id: str
    family_key: str
    proposal_type: str
    trigger_problem_ids: tuple[str, ...]
    control_problem_ids: tuple[str, ...]
    trigger_outcomes: dict[str, int]
    control_outcomes: dict[str, int]
    baseline_variant: str
    candidate_variant: str
    source_judgment_path: str
    design_flags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PreLiveScreenDecision:
    proposal_id: str
    decision: str
    would_run_live: bool
    risk_score: float
    predicted_failure_modes: list[str]
    rationale: str
    feature_trace: dict

    def to_dict(self) -> dict:
        return asdict(self)


def build_pre_live_tie_screen_payload(
    *,
    root: Path,
    cases: Iterable[PreLiveReplayCase] | None = None,
    eval_id: str = "pre_live_tie_screen_20260609",
    benefit_lcb90: float = 0.54,
    control_loss_ucb90: float = 0.35,
) -> dict:
    """Replay a chronological pre-live budget screen over descendant results."""

    root = root.resolve()
    replay_cases = sorted(list(cases or _default_replay_cases()), key=lambda c: c.order)
    history: list[dict] = []
    rows = []
    for case in replay_cases:
        observed = _observed_metrics(case, benefit_lcb90=benefit_lcb90, control_loss_ucb90=control_loss_ucb90)
        screen = _screen_case(case, history)
        deployment_effect = _deployment_effect(screen, observed)
        rows.append({
            "case": {
                **case.to_dict(),
                "source_judgment_path": _display_path(root, root / case.source_judgment_path),
            },
            "observed": observed,
            "screen": screen.to_dict(),
            "deployment_effect": deployment_effect,
        })
        history.append(_history_event(case=case, observed=observed, screen=screen))

    chronological = _chronological_metrics(rows)
    no_screen = _no_screen_metrics(rows)
    retrospective = _retrospective_failure_signature(rows, benefit_lcb90=benefit_lcb90)
    gates = {
        "positive_control_not_blocked": chronological["accepted_positive_block_count"] == 0,
        "positive_control_allowed": chronological["positive_control_allowed_count"] >= 1,
        "failed_live_calls_saved": chronological["failed_live_calls_saved"] >= 4,
        "live_call_reduction_above_half": chronological["live_call_reduction"] >= 0.5,
        "acceptance_rate_improves_among_run_calls": (
            chronological["accepted_rate_among_run_calls"] > no_screen["accepted_rate"]
        ),
        "retrospective_signature_catches_all_failed": (
            retrospective["failed_replay_rows_flagged"] == retrospective["failed_replay_rows"]
        ),
        "main_graph_not_mutated": True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "pre_live_tie_low_benefit_budget_screen",
        "performance_validation": True,
        "validation_scope": (
            "Chronological replay over the orthogonal descendant live line.  The screen sees only earlier "
            "sibling/descendant outcomes or earlier screen deferrals before deciding whether the next "
            "candidate should spend live calls."
        ),
        "thresholds": {
            "benefit_lcb90": benefit_lcb90,
            "control_loss_ucb90": control_loss_ucb90,
            "low_lcb_threshold": 0.40,
            "low_utility_threshold": 0.45,
            "high_overlap_threshold": 0.75,
        },
        "metrics": {
            "no_screen": no_screen,
            "chronological": chronological,
            "retrospective_failure_signature": retrospective,
        },
        "rows": rows,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The accepted seed remains live-runnable, while four of six later rejected live validations are "
            "deferred or blocked before spending.  This is a budget/productivity improvement rather than a "
            "claim that answer quality improves without further ablation."
        ),
    }


def _screen_case(case: PreLiveReplayCase, history: list[dict]) -> PreLiveScreenDecision:
    trigger_set = set(case.trigger_problem_ids)
    family_events = [event for event in history if event["family_key"] == case.family_key]
    same_candidate_events = [
        event for event in history
        if event["candidate_node_id"] == case.candidate_node_id
    ]
    failed_family = [event for event in family_events if event["outcome_class"] in {"reject", "screen_block"}]
    underpowered_family = [
        event for event in family_events
        if event["outcome_class"] in {"underpowered_reject", "screen_defer"}
    ]
    low_benefit_family = [
        event for event in family_events
        if event["observed_trigger_utility"] is not None
        and event["observed_trigger_utility"] <= 0.45
    ]

    max_failed_overlap = max(
        [_overlap(trigger_set, event["trigger_problem_ids"]) for event in failed_family] or [0.0]
    )
    max_low_benefit_overlap = max(
        [_overlap(trigger_set, event["trigger_problem_ids"]) for event in low_benefit_family] or [0.0]
    )
    subset_of_underpowered = any(
        trigger_set.issubset(set(event["trigger_problem_ids"]))
        for event in underpowered_family
    )
    small_trigger_set = len(trigger_set) <= 3

    modes: list[str] = []
    risk = 0.0
    if same_candidate_events:
        modes.append("repeat_candidate_after_prior_reject_or_deferral")
        risk += 0.46
    if max_low_benefit_overlap >= 0.75:
        modes.append("high_overlap_with_low_utility_sibling")
        risk += 0.40
    if subset_of_underpowered and small_trigger_set:
        modes.append("narrower_than_underpowered_sibling")
        risk += 0.34
    if max_failed_overlap >= 0.75 and small_trigger_set:
        modes.append("small_scope_reuses_failed_trigger_set")
        risk += 0.24
    if "scope_only_child" in case.design_flags and family_events:
        modes.append("scope_only_child_after_family_evidence")
        risk += 0.10

    feature_trace = {
        "trigger_count": len(trigger_set),
        "control_count": len(case.control_problem_ids),
        "prior_family_event_count": len(family_events),
        "prior_failed_family_event_count": len(failed_family),
        "prior_underpowered_family_event_count": len(underpowered_family),
        "same_candidate_prior_event_count": len(same_candidate_events),
        "max_failed_trigger_overlap": round(max_failed_overlap, 4),
        "max_low_benefit_trigger_overlap": round(max_low_benefit_overlap, 4),
        "subset_of_underpowered_sibling": subset_of_underpowered,
        "small_trigger_set": small_trigger_set,
        "design_flags": list(case.design_flags),
    }

    if same_candidate_events:
        decision = "block_repeat_rejected_candidate"
        would_run = False
        rationale = "The same candidate already failed or was deferred in this family."
    elif max_low_benefit_overlap >= 0.75:
        decision = "block_predicted_low_benefit"
        would_run = False
        rationale = "The trigger set strongly overlaps a prior low-utility sibling."
    elif subset_of_underpowered and small_trigger_set:
        decision = "defer_expand_before_live"
        would_run = False
        rationale = "A narrower child of an underpowered sibling should expand evidence before live spend."
    elif max_failed_overlap >= 0.75 and small_trigger_set:
        decision = "defer_expand_before_live"
        would_run = False
        rationale = "The small trigger set mostly reuses a failed sibling scope."
    else:
        decision = "run_live"
        would_run = True
        rationale = "No prior sibling evidence predicts a repeat tie or low-benefit result."

    return PreLiveScreenDecision(
        proposal_id=case.proposal_id,
        decision=decision,
        would_run_live=would_run,
        risk_score=round(min(1.0, risk), 4),
        predicted_failure_modes=sorted(set(modes)),
        rationale=rationale,
        feature_trace=feature_trace,
    )


def _observed_metrics(
    case: PreLiveReplayCase,
    *,
    benefit_lcb90: float,
    control_loss_ucb90: float,
) -> dict:
    trigger_n = sum(case.trigger_outcomes.values())
    control_n = sum(case.control_outcomes.values())
    trigger_utility = _utility(case.trigger_outcomes)
    trigger_lcb = _normal_bound(trigger_utility, trigger_n, sign=-1) if trigger_n else None
    control_loss_rate = case.control_outcomes.get("loss", 0) / control_n if control_n else None
    control_loss_ucb = _normal_bound(control_loss_rate, control_n, sign=1) if control_loss_rate is not None else None
    if trigger_lcb is None:
        observed_decision = "insufficient_judgments"
    elif trigger_lcb < benefit_lcb90:
        observed_decision = "reject_benefit"
    elif control_loss_ucb is not None and control_loss_ucb > control_loss_ucb90:
        observed_decision = "reject_harm"
    else:
        observed_decision = "accept"
    return {
        "decision": observed_decision,
        "trigger_outcomes": dict(case.trigger_outcomes),
        "control_outcomes": dict(case.control_outcomes),
        "trigger_n": trigger_n,
        "control_n": control_n,
        "trigger_utility": None if trigger_lcb is None else round(trigger_utility, 4),
        "trigger_lcb90": None if trigger_lcb is None else round(trigger_lcb, 4),
        "trigger_tie_rate": (
            round(case.trigger_outcomes.get("tie", 0) / trigger_n, 4) if trigger_n else None
        ),
        "control_loss_rate": None if control_loss_rate is None else round(control_loss_rate, 4),
        "control_loss_ucb90": None if control_loss_ucb is None else round(control_loss_ucb, 4),
        "outcome_class": _outcome_class(observed_decision, trigger_utility, trigger_lcb),
    }


def _deployment_effect(screen: PreLiveScreenDecision, observed: dict) -> dict:
    accepted = observed["decision"] == "accept"
    would_run = screen.would_run_live
    return {
        "live_call_spent": would_run,
        "failed_live_call_saved": (not would_run) and not accepted,
        "accepted_candidate_blocked": (not would_run) and accepted,
        "failed_live_call_still_spent": would_run and not accepted,
        "accepted_live_call_preserved": would_run and accepted,
    }


def _history_event(
    *,
    case: PreLiveReplayCase,
    observed: dict,
    screen: PreLiveScreenDecision,
) -> dict:
    if screen.would_run_live:
        outcome_class = observed["outcome_class"]
        observed_utility = observed["trigger_utility"]
        observed_lcb = observed["trigger_lcb90"]
    elif screen.decision.startswith("defer"):
        outcome_class = "screen_defer"
        observed_utility = None
        observed_lcb = None
    else:
        outcome_class = "screen_block"
        observed_utility = None
        observed_lcb = None
    return {
        "proposal_id": case.proposal_id,
        "candidate_node_id": case.candidate_node_id,
        "family_key": case.family_key,
        "trigger_problem_ids": tuple(case.trigger_problem_ids),
        "screen_decision": screen.decision,
        "outcome_class": outcome_class,
        "observed_trigger_utility": observed_utility,
        "observed_trigger_lcb90": observed_lcb,
    }


def _chronological_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    run_rows = [row for row in rows if row["deployment_effect"]["live_call_spent"]]
    blocked_rows = [row for row in rows if not row["deployment_effect"]["live_call_spent"]]
    accepted_rows = [row for row in rows if row["observed"]["decision"] == "accept"]
    failed_rows = [row for row in rows if row["observed"]["decision"] != "accept"]
    saved_failed = [row for row in rows if row["deployment_effect"]["failed_live_call_saved"]]
    blocked_accepted = [row for row in rows if row["deployment_effect"]["accepted_candidate_blocked"]]
    preserved_accepted = [row for row in rows if row["deployment_effect"]["accepted_live_call_preserved"]]
    failed_spent = [row for row in rows if row["deployment_effect"]["failed_live_call_still_spent"]]
    return {
        "case_count": total,
        "live_calls_with_screen": len(run_rows),
        "blocked_or_deferred_count": len(blocked_rows),
        "accepted_observed_count": len(accepted_rows),
        "failed_observed_count": len(failed_rows),
        "failed_live_calls_saved": len(saved_failed),
        "failed_live_calls_still_spent": len(failed_spent),
        "accepted_positive_block_count": len(blocked_accepted),
        "positive_control_allowed_count": len(preserved_accepted),
        "live_call_reduction": round((total - len(run_rows)) / total, 4) if total else 0.0,
        "accepted_rate_among_run_calls": (
            round(len(preserved_accepted) / len(run_rows), 4) if run_rows else 0.0
        ),
        "failed_saved_proposal_ids": [row["case"]["proposal_id"] for row in saved_failed],
        "failed_spent_proposal_ids": [row["case"]["proposal_id"] for row in failed_spent],
        "blocked_accepted_proposal_ids": [row["case"]["proposal_id"] for row in blocked_accepted],
        "decision_counts": dict(Counter(row["screen"]["decision"] for row in rows)),
    }


def _no_screen_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    accepted = sum(1 for row in rows if row["observed"]["decision"] == "accept")
    failed = total - accepted
    return {
        "live_calls": total,
        "accepted_count": accepted,
        "failed_count": failed,
        "accepted_rate": round(accepted / total, 4) if total else 0.0,
        "decision_counts": dict(Counter(row["observed"]["decision"] for row in rows)),
    }


def _retrospective_failure_signature(rows: list[dict], *, benefit_lcb90: float) -> dict:
    failed = [row for row in rows if row["observed"]["decision"] != "accept"]
    flagged = []
    for row in failed:
        observed = row["observed"]
        low_lcb = observed["trigger_lcb90"] is not None and observed["trigger_lcb90"] < benefit_lcb90
        low_utility = observed["trigger_utility"] is not None and observed["trigger_utility"] <= 0.50
        high_tie = observed["trigger_tie_rate"] is not None and observed["trigger_tie_rate"] >= 0.50
        if low_lcb or low_utility or high_tie:
            flagged.append(row)
    return {
        "failed_replay_rows": len(failed),
        "failed_replay_rows_flagged": len(flagged),
        "flagged_failed_proposal_ids": [row["case"]["proposal_id"] for row in flagged],
        "signature": "trigger_lcb90_below_gate OR trigger_utility<=0.50 OR trigger_tie_rate>=0.50",
    }


def _outcome_class(decision: str, trigger_utility: float, trigger_lcb: float | None) -> str:
    if decision == "accept":
        return "accept"
    if decision == "reject_harm":
        return "reject"
    if trigger_lcb is not None and trigger_lcb < 0.40 and trigger_utility > 0.55:
        return "underpowered_reject"
    return "reject"


def _overlap(left: set[str], right: Iterable[str]) -> float:
    right_set = set(right)
    if not left:
        return 0.0
    return len(left & right_set) / len(left)


def _utility(outcomes: dict[str, int]) -> float:
    n = sum(outcomes.values())
    return (
        (outcomes.get("win", 0) + 0.5 * outcomes.get("tie", 0)) / n
        if n
        else 0.0
    )


def _normal_bound(value: float, n: int, *, sign: int) -> float:
    if n <= 0:
        return value
    se = math.sqrt(max(value * (1.0 - value), 0.0) / n)
    return max(0.0, min(1.0, value + sign * 1.28 * se))


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _default_replay_cases() -> list[PreLiveReplayCase]:
    parent_triggers = (
        "business_0097",
        "business_0192",
        "software_engineering_0142",
        "business_0218",
        "daily_life_0173",
    )
    business_triggers = (
        "business_0097",
        "business_0192",
        "business_0218",
        "daily_life_0173",
    )
    technical_triggers = (
        "engineering_0244",
        "software_engineering_0142",
        "software_engineering_0265",
        "software_engineering_0337",
        "software_engineering_0379",
    )
    retained_technical_triggers = (
        "engineering_0244",
        "software_engineering_0142",
        "software_engineering_0379",
    )
    business_controls = (
        "daily_life_0183",
        "engineering_0244",
        "engineering_0183",
        "business_0146",
        "engineering_0152",
        "daily_life_0197",
        "software_engineering_0142",
        "software_engineering_0265",
    )
    technical_controls = (
        "business_0097",
        "daily_life_0183",
        "engineering_0183",
        "business_0146",
        "engineering_0152",
        "business_0192",
        "daily_life_0197",
        "software_engineering_0265",
    )
    return [
        PreLiveReplayCase(
            order=1,
            proposal_id="prop_d7abf65010d2",
            candidate_node_id="cand_f8ca2582dbc4",
            parent_node_id="cand_39de0aeae8a3",
            family_key="orthogonal_execution_descendant",
            proposal_type="descendant_execution_hypothesis",
            trigger_problem_ids=parent_triggers,
            control_problem_ids=(
                "daily_life_0183",
                "engineering_0244",
                "engineering_0183",
                "business_0146",
                "engineering_0152",
                "daily_life_0197",
                "software_engineering_0265",
                "daily_life_0216",
            ),
            trigger_outcomes={"win": 4, "loss": 1, "tie": 0},
            control_outcomes={"tie": 8, "win": 0, "loss": 0},
            baseline_variant="phase2_v20_claude_opus_execution_baseline",
            candidate_variant="proposal_d7abf65010d2",
            source_judgment_path=(
                "phase two/analysis/cache/judgments/"
                "proposal_d7abf65010d2_vs_phase2_v20_claude_opus_execution_baseline_"
                "orthogonal_descendant_live_same_model_20260608_prop_d7abf65010d2.json"
            ),
            design_flags=("accepted_seed",),
        ),
        PreLiveReplayCase(
            order=2,
            proposal_id="prop_d44aae0f9127",
            candidate_node_id="cand_e1fb6b49c911",
            parent_node_id="cand_f8ca2582dbc4",
            family_key="orthogonal_execution_descendant",
            proposal_type="descendant_execution_hypothesis",
            trigger_problem_ids=business_triggers,
            control_problem_ids=business_controls,
            trigger_outcomes={"win": 1, "loss": 3, "tie": 0},
            control_outcomes={"tie": 8, "win": 0, "loss": 0},
            baseline_variant="phase2_v20_claude_opus_execution_baseline",
            candidate_variant="proposal_d44aae0f9127",
            source_judgment_path=(
                "phase two/analysis/cache/judgments/"
                "proposal_d44aae0f9127_vs_phase2_v20_claude_opus_execution_baseline_"
                "orthogonal_descendant_nextgen_live_same_model_20260609_prop_d44aae0f9127.json"
            ),
            design_flags=("scope_only_child",),
        ),
        PreLiveReplayCase(
            order=3,
            proposal_id="prop_99b7c2f9b052",
            candidate_node_id="cand_d06917ca25b2",
            parent_node_id="cand_f8ca2582dbc4",
            family_key="orthogonal_execution_descendant",
            proposal_type="descendant_execution_hypothesis",
            trigger_problem_ids=business_triggers,
            control_problem_ids=business_controls,
            trigger_outcomes={"win": 1, "loss": 2, "tie": 1},
            control_outcomes={"tie": 8, "win": 0, "loss": 0},
            baseline_variant="phase2_v20_claude_opus_execution_baseline",
            candidate_variant="proposal_99b7c2f9b052",
            source_judgment_path=(
                "phase two/analysis/cache/judgments/"
                "proposal_99b7c2f9b052_vs_phase2_v20_claude_opus_execution_baseline_"
                "orthogonal_descendant_nextgen_live_same_model_20260609_prop_99b7c2f9b052.json"
            ),
            design_flags=("residual_specific_repair", "same_scope_as_failed_sibling"),
        ),
        PreLiveReplayCase(
            order=4,
            proposal_id="prop_584773b088ff",
            candidate_node_id="cand_9e1f0a40978e",
            parent_node_id="cand_f8ca2582dbc4",
            family_key="orthogonal_execution_descendant",
            proposal_type="descendant_execution_hypothesis",
            trigger_problem_ids=technical_triggers,
            control_problem_ids=technical_controls,
            trigger_outcomes={"win": 3, "loss": 1, "tie": 1},
            control_outcomes={"tie": 8, "win": 0, "loss": 0},
            baseline_variant="phase2_v20_claude_opus_technical_baseline",
            candidate_variant="proposal_584773b088ff",
            source_judgment_path=(
                "phase two/analysis/cache/judgments/"
                "proposal_584773b088ff_vs_phase2_v20_claude_opus_technical_baseline_"
                "orthogonal_technical_descendant_live_same_model_20260609_prop_584773b088ff.json"
            ),
            design_flags=("new_technical_cluster",),
        ),
        PreLiveReplayCase(
            order=5,
            proposal_id="prop_412034c92b89",
            candidate_node_id="cand_30cce6f5db89",
            parent_node_id="cand_f8ca2582dbc4",
            family_key="orthogonal_execution_descendant",
            proposal_type="descendant_execution_hypothesis",
            trigger_problem_ids=retained_technical_triggers,
            control_problem_ids=technical_controls,
            trigger_outcomes={"win": 1, "loss": 1, "tie": 1},
            control_outcomes={"tie": 8, "win": 0, "loss": 0},
            baseline_variant="phase2_v20_claude_opus_technical_baseline",
            candidate_variant="proposal_412034c92b89",
            source_judgment_path=(
                "phase two/analysis/cache/judgments/"
                "proposal_412034c92b89_vs_phase2_v20_claude_opus_technical_baseline_"
                "orthogonal_technical_descendant_live_same_model_20260609_prop_412034c92b89.json"
            ),
            design_flags=("retained_subset_child",),
        ),
        PreLiveReplayCase(
            order=6,
            proposal_id="prop_6c22137d982d",
            candidate_node_id="cand_042b28bf889c",
            parent_node_id="cand_f8ca2582dbc4",
            family_key="orthogonal_execution_descendant",
            proposal_type="descendant_execution_hypothesis",
            trigger_problem_ids=retained_technical_triggers,
            control_problem_ids=technical_controls,
            trigger_outcomes={"win": 1, "loss": 0, "tie": 2},
            control_outcomes={"tie": 8, "win": 0, "loss": 0},
            baseline_variant="phase2_v20_claude_opus_technical_baseline",
            candidate_variant="proposal_6c22137d982d",
            source_judgment_path=(
                "phase two/analysis/cache/judgments/"
                "proposal_6c22137d982d_vs_phase2_v20_claude_opus_technical_baseline_"
                "orthogonal_technical_descendant_live_same_model_20260609_prop_6c22137d982d.json"
            ),
            design_flags=("residual_specific_repair", "same_scope_as_deferred_sibling"),
        ),
        PreLiveReplayCase(
            order=7,
            proposal_id="prop_6c22137d982d_vs_parent",
            candidate_node_id="cand_042b28bf889c",
            parent_node_id="cand_f8ca2582dbc4",
            family_key="orthogonal_execution_descendant",
            proposal_type="descendant_execution_hypothesis_parent_comparator",
            trigger_problem_ids=retained_technical_triggers,
            control_problem_ids=technical_controls,
            trigger_outcomes={"win": 0, "loss": 1, "tie": 2},
            control_outcomes={"tie": 8, "win": 0, "loss": 0},
            baseline_variant="proposal_d7abf65010d2_technical_parent",
            candidate_variant="proposal_6c22137d982d",
            source_judgment_path=(
                "phase two/analysis/cache/judgments/"
                "proposal_6c22137d982d_vs_proposal_d7abf65010d2_technical_parent_"
                "orthogonal_technical_descendant_vs_parent_live_20260609_prop_6c22137d982d.json"
            ),
            design_flags=("repeat_candidate_parent_comparator",),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pre-live tie/low-benefit screen replay payload.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="pre_live_tie_screen_20260609")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_pre_live_tie_screen_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"]["chronological"],
        "out": _display_path(root, out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
