"""V2 process-model zoo and process-family alignment benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from .hypothesis_lifecycle_v2 import AlignmentHypothesis, ProcessModel
from .schema import stable_id


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "process_model_zoo_v2_20260610.json"


@dataclass(frozen=True)
class ProcessZooEntry:
    model: ProcessModel
    family_tags: tuple[str, ...]
    role_schema: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "family_tags": list(self.family_tags),
            "role_schema": list(self.role_schema),
        }


@dataclass(frozen=True)
class PairJudgment:
    source_id: str
    target_id: str
    score: float
    decision: str
    gold_label: str
    family_overlap: list[str]
    role_overlap: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_process_model_zoo_v2_payload(
    *,
    eval_id: str = "process_model_zoo_v2_20260610",
) -> dict[str, Any]:
    entries = process_zoo_entries()
    entry_by_id = {entry.model.id: entry for entry in entries}
    positive_pairs = _gold_positive_pairs()
    negative_pairs = _gold_negative_pairs()
    judgments = [
        judge_process_pair(entry_by_id[left], entry_by_id[right], gold_label="positive")
        for left, right in sorted(positive_pairs)
    ] + [
        judge_process_pair(entry_by_id[left], entry_by_id[right], gold_label="negative")
        for left, right in sorted(negative_pairs)
    ]
    confusion = _confusion(judgments)
    alignments = [
        alignment_from_judgment(entry_by_id, row)
        for row in judgments
        if row.gold_label == "positive" and row.decision == "align"
    ]
    validation_issues = {
        entry.model.id: entry.model.validate()
        for entry in entries
    }
    family_counts: dict[str, int] = {}
    for entry in entries:
        for family in entry.family_tags:
            family_counts[family] = family_counts.get(family, 0) + 1
    metrics = {
        "process_count": len(entries),
        "family_count": len(family_counts),
        "positive_pair_count": len(positive_pairs),
        "negative_pair_count": len(negative_pairs),
        "alignment_hypothesis_count": len(alignments),
        "validation_issue_count": sum(len(v) for v in validation_issues.values()),
        **confusion,
    }
    gates = {
        "has_ten_process_models": metrics["process_count"] == 10,
        "all_process_models_validate": metrics["validation_issue_count"] == 0,
        "has_multiple_process_families": metrics["family_count"] >= 5,
        "positive_alignment_recall_high": metrics["positive_recall"] >= 0.85,
        "negative_control_rejection_high": metrics["negative_rejection_rate"] >= 0.85,
        "overall_pair_accuracy_high": metrics["accuracy"] >= 0.85,
        "alignment_nodes_available_for_positives": metrics["alignment_hypothesis_count"] >= 6,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "process_model_zoo_v2_alignment_benchmark",
        "reconstruction_v2_phase": "phase2_process_model_zoo",
        "performance_validation": True,
        "validation_scope": (
            "Ten typed process models plus gold positive/negative process-family alignment pairs. "
            "Performance is measured as deterministic alignment classification over held fixture pairs."
        ),
        "process_entries": [entry.to_dict() for entry in entries],
        "family_counts": family_counts,
        "gold_positive_pairs": [list(pair) for pair in sorted(positive_pairs)],
        "gold_negative_pairs": [list(pair) for pair in sorted(negative_pairs)],
        "pair_judgments": [row.to_dict() for row in judgments],
        "alignment_hypotheses": [row.to_dict() for row in alignments],
        "validation_issues": validation_issues,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Phase 2 gives the v2 agent a structured process state space.  Alignment is now tested between "
            "typed process models and families, not between opaque principle names."
        ),
    }


def process_zoo_entries() -> list[ProcessZooEntry]:
    return [
        _entry(
            "process_le_chatelier_v1",
            "chemical_thermodynamic_equilibrium",
            ["reaction_quotient", "composition", "temperature", "pressure"],
            ["external_condition_change"],
            "imposed external condition change",
            "equilibrium shift partially counteracts the imposed change",
            ["response opposes perturbation", "moves toward constrained equilibrium"],
            ["far from equilibrium", "irreversible reaction"],
            ("negative_feedback", "equilibrium_restoration"),
            ("perturbation", "opposing_response", "constraint"),
        ),
        _entry(
            "process_lenz_law_v1",
            "electromagnetic_induction",
            ["magnetic_flux", "induced_emf", "induced_current"],
            ["external_flux_change"],
            "change in magnetic flux over time",
            "induced current creates field opposing the flux change",
            ["response opposes perturbation", "sign-sensitive local opposition"],
            ["open circuit", "nonlinear magnetic material"],
            ("negative_feedback", "opposition_response"),
            ("perturbation", "opposing_response", "sign_relation"),
        ),
        _entry(
            "process_thermostat_v1",
            "thermostat_temperature_control",
            ["temperature", "setpoint", "heating_power"],
            ["external_heat_loss", "setpoint_change"],
            "temperature deviates from setpoint",
            "controller changes heating or cooling to reduce deviation",
            ["response opposes deviation", "closed-loop error reduction"],
            ["sensor failure", "actuator saturation"],
            ("negative_feedback", "control_loop", "equilibrium_restoration"),
            ("perturbation", "opposing_response", "error_signal"),
        ),
        _entry(
            "process_predator_prey_local_v1",
            "predator_prey_local_stabilization",
            ["prey_population", "predator_population", "growth_rate"],
            ["prey_shock", "predator_shock"],
            "population perturbation",
            "coupled response can restore local balance near fixed point",
            ["coupled variables can damp local deviation", "stability depends on parameters"],
            ["limit cycle regime", "parameter instability"],
            ("coupled_dynamics", "local_stabilization"),
            ("state_coupling", "restoring_response", "fixed_point"),
        ),
        _entry(
            "process_first_order_decay_v1",
            "chemical_first_order_decay",
            ["concentration", "rate_constant", "time"],
            ["initial_concentration_change"],
            "initial amount is set away from zero",
            "quantity decreases proportionally to current amount",
            ["monotone exponential decay", "rate proportional to state"],
            ["multi-step reaction", "reversible reaction"],
            ("exponential_decay", "monotone_relaxation"),
            ("state", "rate_proportional_to_state", "monotone_decrease"),
        ),
        _entry(
            "process_radioactive_decay_v1",
            "radioactive_decay",
            ["nuclei_count", "decay_constant", "time"],
            ["initial_nuclei_count"],
            "initial unstable nuclei count",
            "expected count decays proportionally to current count",
            ["monotone exponential decay", "memoryless decay law"],
            ["daughter-chain coupling", "measurement noise"],
            ("exponential_decay", "memoryless_process"),
            ("state", "rate_proportional_to_state", "monotone_decrease"),
        ),
        _entry(
            "process_rc_discharge_v1",
            "rc_circuit_discharge",
            ["capacitor_voltage", "resistance", "capacitance", "time"],
            ["initial_voltage"],
            "capacitor starts charged above equilibrium",
            "voltage decays proportionally to current voltage",
            ["monotone exponential decay", "time constant controls relaxation"],
            ["nonlinear component", "driven circuit"],
            ("exponential_decay", "monotone_relaxation"),
            ("state", "rate_proportional_to_state", "monotone_decrease"),
        ),
        _entry(
            "process_logistic_growth_v1",
            "logistic_growth",
            ["population", "carrying_capacity", "growth_rate"],
            ["initial_population"],
            "population starts below or above carrying capacity",
            "growth slows as population approaches carrying capacity",
            ["saturation near carrying capacity", "state-dependent growth slowdown"],
            ["changing carrying capacity", "strong stochastic shocks"],
            ("saturation", "capacity_constraint"),
            ("state", "capacity_limit", "growth_slowdown"),
        ),
        _entry(
            "process_damped_oscillator_v1",
            "damped_oscillator",
            ["position", "velocity", "damping", "spring_constant"],
            ["initial_displacement", "initial_velocity"],
            "system displaced from equilibrium",
            "restoring force and damping reduce oscillation energy",
            ["restoring force opposes displacement", "energy dissipates over time"],
            ["negative damping", "nonlinear forcing"],
            ("oscillation", "damping", "local_stabilization"),
            ("restoring_response", "energy_dissipation", "trajectory_shape"),
        ),
        _entry(
            "process_supply_demand_response_v1",
            "supply_demand_equilibrium_response",
            ["price", "demand", "supply", "inventory"],
            ["demand_shock", "supply_shock"],
            "market shock moves price or quantity away from clearing level",
            "price and quantity adjust toward local clearing balance",
            ["adjustment can reduce shortage or surplus", "equilibrium restoration is local and delayed"],
            ["sticky prices", "market power", "external rationing"],
            ("equilibrium_restoration", "negative_feedback", "market_adjustment"),
            ("perturbation", "opposing_response", "constraint"),
        ),
    ]


def judge_process_pair(left: ProcessZooEntry, right: ProcessZooEntry, *, gold_label: str) -> PairJudgment:
    family_overlap = sorted(set(left.family_tags) & set(right.family_tags))
    role_overlap = sorted(set(left.role_schema) & set(right.role_schema))
    family_score = len(family_overlap) / max(1, min(len(left.family_tags), len(right.family_tags)))
    role_score = len(role_overlap) / max(1, min(len(left.role_schema), len(right.role_schema)))
    invariant_score = _token_jaccard(
        " ".join(left.model.invariants),
        " ".join(right.model.invariants),
    )
    score = round(0.55 * family_score + 0.30 * role_score + 0.15 * invariant_score, 4)
    decision = "align" if score >= 0.45 and family_overlap and len(role_overlap) >= 2 else "reject"
    rationale = (
        f"family_overlap={family_overlap}; role_overlap={role_overlap}; "
        f"invariant_score={round(invariant_score, 4)}"
    )
    return PairJudgment(
        source_id=left.model.id,
        target_id=right.model.id,
        score=score,
        decision=decision,
        gold_label=gold_label,
        family_overlap=family_overlap,
        role_overlap=role_overlap,
        rationale=rationale,
    )


def alignment_from_judgment(entry_by_id: dict[str, ProcessZooEntry], judgment: PairJudgment) -> AlignmentHypothesis:
    left = entry_by_id[judgment.source_id]
    right = entry_by_id[judgment.target_id]
    return AlignmentHypothesis(
        id=stable_id("align2", judgment.source_id, judgment.target_id),
        source_process=judgment.source_id,
        target_process=judgment.target_id,
        mapping={
            "shared_family": " / ".join(judgment.family_overlap),
            "shared_roles": " / ".join(judgment.role_overlap),
            "source_perturbation": left.model.perturbation,
            "target_perturbation": right.model.perturbation,
            "source_response": left.model.response,
            "target_response": right.model.response,
        },
        preserved_structure=[
            f"shared family: {', '.join(judgment.family_overlap)}",
            f"shared roles: {', '.join(judgment.role_overlap)}",
        ],
        broken_structure=[
            f"domain differs: {left.model.domain} vs {right.model.domain}",
            "equations and state variables are not assumed identical",
        ],
        metric_scores={
            "process_family_score": judgment.score,
        },
        verifier_tests=[
            "typed_process_pair_check",
            "negative_control_pair_check",
            "fresh_transfer_task_ablation",
        ],
    )


def _entry(
    pid: str,
    domain: str,
    state_variables: list[str],
    interventions: list[str],
    perturbation: str,
    response: str,
    invariants: list[str],
    failure_cases: list[str],
    family_tags: tuple[str, ...],
    role_schema: tuple[str, ...],
) -> ProcessZooEntry:
    model = ProcessModel(
        id=pid,
        domain=domain,
        state_variables=state_variables,
        parameters=["domain_parameters"],
        interventions=interventions,
        perturbation=perturbation,
        response=response,
        dynamics={
            "family_tags": list(family_tags),
            "role_schema": list(role_schema),
        },
        observation_map="observe process state before and after intervention",
        invariants=invariants,
        failure_cases=failure_cases,
    )
    return ProcessZooEntry(model=model, family_tags=family_tags, role_schema=role_schema)


def _gold_positive_pairs() -> set[tuple[str, str]]:
    return _pairs({
        ("process_le_chatelier_v1", "process_lenz_law_v1"),
        ("process_le_chatelier_v1", "process_thermostat_v1"),
        ("process_le_chatelier_v1", "process_supply_demand_response_v1"),
        ("process_lenz_law_v1", "process_thermostat_v1"),
        ("process_thermostat_v1", "process_supply_demand_response_v1"),
        ("process_first_order_decay_v1", "process_radioactive_decay_v1"),
        ("process_first_order_decay_v1", "process_rc_discharge_v1"),
        ("process_radioactive_decay_v1", "process_rc_discharge_v1"),
        ("process_predator_prey_local_v1", "process_damped_oscillator_v1"),
    })


def _gold_negative_pairs() -> set[tuple[str, str]]:
    return _pairs({
        ("process_lenz_law_v1", "process_radioactive_decay_v1"),
        ("process_le_chatelier_v1", "process_first_order_decay_v1"),
        ("process_logistic_growth_v1", "process_rc_discharge_v1"),
        ("process_damped_oscillator_v1", "process_radioactive_decay_v1"),
        ("process_supply_demand_response_v1", "process_first_order_decay_v1"),
        ("process_predator_prey_local_v1", "process_rc_discharge_v1"),
        ("process_logistic_growth_v1", "process_lenz_law_v1"),
    })


def _pairs(raw: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return {tuple(sorted(pair)) for pair in raw}


def _confusion(judgments: list[PairJudgment]) -> dict[str, Any]:
    tp = sum(1 for row in judgments if row.gold_label == "positive" and row.decision == "align")
    fn = sum(1 for row in judgments if row.gold_label == "positive" and row.decision != "align")
    tn = sum(1 for row in judgments if row.gold_label == "negative" and row.decision == "reject")
    fp = sum(1 for row in judgments if row.gold_label == "negative" and row.decision != "reject")
    total = len(judgments)
    return {
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "positive_recall": round(tp / max(1, tp + fn), 4),
        "positive_precision": round(tp / max(1, tp + fp), 4),
        "negative_rejection_rate": round(tn / max(1, tn + fp), 4),
    }


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if len(token) > 2
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 process model zoo alignment benchmark.")
    parser.add_argument("--eval-id", default="process_model_zoo_v2_20260610")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_process_model_zoo_v2_payload(eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
