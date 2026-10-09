#!/usr/bin/env python3
"""Independent-mechanism retain/delete comparison for source feedback."""

from copy import deepcopy
from dataclasses import replace
from math import isfinite
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "source_feedback_random_retention_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from run_source_feedback_task_family import ParametricConflictLab, _fixtures, _offers, _previous


SEED = 20261004
RANDOM_TASK_COUNT = 8


class SessionLab(ParametricConflictLab):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _config():
    return ResearchConfig(observation_budget=100, compute_budget=20_000,
        minimum_improvement=1e-7, max_rounds=1, active_probes=False,
        risk_learning=False, seed=SEED, max_candidates=2)


def _task_variants():
    """Draw mechanisms independently; include a repeated compatible anchor."""
    rng = Random(SEED)
    variants = [{"label": "anchor_base_0", "current_weight": .55,
                 "lag_weight": 1.2, "noise": 0.}]
    for index in range(RANDOM_TASK_COUNT):
        variants.append({
            "label": f"random_mechanism_{index}",
            "current_weight": round(rng.uniform(.25, .9), 3),
            "lag_weight": round(rng.uniform(.55, 1.8), 3),
            "noise": rng.choice((0., .01, .02, .04)),
        })
    variants.append({"label": "anchor_base_1", "current_weight": .55,
                     "lag_weight": 1.2, "noise": 0.})
    return variants


def _prepare(case, good, library, variants):
    tasks, specs, pending, allocation_limits = [], [], [], {}
    for position, variant in enumerate(variants):
        spec = replace(case["spec"], task_id=f"random_retention_{position}_{variant['label']}")
        lab = ParametricConflictLab(spec, current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"],
            seed=SEED + position)
        previous = _previous(case, lab)
        config, offers = _offers(case, good, library, lab, previous, SEED + position)
        if len({tuple(sorted(offer["minimum_resources"].items())) for offer in offers}) != 1:
            raise RuntimeError(f"source offers disagree on fixed allocation at task {position}")
        allocation_limits[position] = dict(offers[0]["minimum_resources"])
        for offer in offers:
            offer.update(task_position=position, source_task_position=position)
        tasks.append({"task_index": position, "seed": SEED + position,
                      "variant": variant["label"],
                      "research_limits": {"observation_budget": config.observation_budget,
                          "compute_budget": config.compute_budget,
                          "measurement_budget": 0}})
        specs.append((spec, variant))
        pending.extend(offers)
    return tasks, specs, pending, allocation_limits


def _run_arm(case, good, base_library, arm, prepared):
    config = _config()
    library = KnowledgeLibrary.from_dict(deepcopy(base_library.to_dict()))
    tasks, specs, pending, allocation_limits = prepared
    if arm == "matched_delete":
        payload = library.to_dict()
        for concept in payload["concepts"]:
            concept["reuse_history"] = [item for item in concept.get("reuse_history", [])
                if item.get("kind") != "research_return"]
        library = KnowledgeLibrary.from_dict(payload)

    def factory(item):
        spec, variant = specs[item["task_index"]]
        return SessionLab(spec, current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"],
            seed=item["seed"])

    session = UnifiedResearchSession(config, tasks=tasks, library=library,
        measurement_budget=0, lab_factory=factory)
    previous_rows = []
    for position, (spec, variant) in enumerate(specs):
        previous_rows.append({"task_position": position, "research": deepcopy(
            _previous(case, ParametricConflictLab(spec,
                current_weight=variant["current_weight"],
                lag_weight=variant["lag_weight"], noise=variant["noise"],
                seed=SEED + position)))})
    session.state["results"] = previous_rows
    session.state["pending_revisits"] = deepcopy(pending)
    session.state["screening_compute_cost"] = 0
    rows = []
    for position, (spec, variant) in enumerate(specs):
        session._refresh_source_feedback()
        candidates = [item for item in session.state["pending_revisits"]
                      if item["task_position"] == position]
        if not candidates:
            rows.append({"task_position": position, "variant": variant["label"], "arm": arm,
                "transition": {key: variant[key] for key in ("current_weight", "lag_weight", "noise")},
                "selected_source": [], "status": "withdrawn_before_execution",
                "heldout_loss": None, "visit_costs": {"observation_cost": 0,
                    "measurement_cost": 0, "compute_cost": 0, "queries": 0},
                "allocation_limits": dict(allocation_limits[position]),
                "allocation_feasible": None,
                "cumulative_costs": dict(session._costs()),
                "feedback_events": deepcopy(session.state["scheduler_events"]),
                "external_model_calls": 0})
            session.state["pending_revisits"] = [item for item in session.state["pending_revisits"]
                if item["task_position"] > position]
            continue
        chosen = candidates[0]
        limits = dict(allocation_limits[position])
        if any(chosen["minimum_resources"][key] > limits[key] for key in limits):
            raise RuntimeError(f"fixed allocation cannot fund task {position}/{arm}")
        before = session._costs()
        if not session._perform_revisit(chosen, allocation_limits=limits):
            raise RuntimeError(f"source revisit failed at task {position}/{arm}")
        report = session.state["results"][position]["research"]
        after = session._costs()
        heldout_lab = ParametricConflictLab(spec,
            current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"],
            seed=SEED + position + 100_000)
        heldout = heldout_lab.observe(spec.available_probes[-1])
        heldout_loss = loss(Program.from_dict(report["program"]), heldout,
                            report["fallback"], .5)
        if not isfinite(heldout_loss):
            raise RuntimeError(f"non-finite held-out loss at task {position}/{arm}")
        rows.append({"task_position": position, "variant": variant["label"],
            "arm": arm, "transition": {key: variant[key] for key in
                ("current_weight", "lag_weight", "noise")},
            "selected_source": list(chosen["candidate"]["source_concept_ids"]),
            "status": report["status"], "heldout_loss": heldout_loss,
            "visit_costs": dict(report["visit_costs"]),
            "allocation_limits": limits,
            "allocation_feasible": all(report["visit_costs"][key] <= limits[key]
                                        for key in limits),
            "cumulative_costs": {key: after[key] for key in after},
            "feedback_events": deepcopy(session.state["scheduler_events"]),
            "external_model_calls": 0})
        session.state["pending_revisits"] = [item for item in session.state["pending_revisits"]
            if item["task_position"] > position]
        if arm == "matched_delete":
            for concept in session.library.concepts:
                concept.reuse_history = [item for item in concept.reuse_history
                    if item.get("kind") != "research_return"]
    return {"arm": arm, "tasks": rows, "total_costs": session._costs(),
            "pending_after": len(session.state["pending_revisits"]),
            "external_model_calls": 0}


def main():
    case, good, warm, cold = _fixtures()
    variants = _task_variants()
    # Screen once from the common warm library.  The arm difference starts
    # only after the offers and their resource quotes have been frozen.
    prepared = _prepare(case, good, KnowledgeLibrary.from_dict(deepcopy(warm.to_dict())), variants)
    arms = {"feedback": _run_arm(case, good, warm, "feedback", prepared),
            "matched_delete": _run_arm(case, good, cold, "matched_delete", prepared)}
    feedback = arms["feedback"]["tasks"]
    matched = arms["matched_delete"]["tasks"]
    assert len(feedback) == len(matched) == len(variants)
    assert all(row["external_model_calls"] == 0 for row in feedback + matched)
    divergences = [index for index, (left, right) in enumerate(zip(feedback, matched))
                   if left["selected_source"] != right["selected_source"]]
    for left, right in zip(feedback, matched):
        assert left["variant"] == right["variant"]
        assert left["transition"] == right["transition"]
        assert left["allocation_limits"] == right["allocation_limits"]
        assert left["allocation_feasible"] is not False
        assert right["allocation_feasible"] is not False
    payload = {"status": "complete", "lineage": {
        "package_version": "0.181.0", "parent": "source-feedback selection",
        "stage": "independent random mechanisms with source retain/delete"},
        "protocol": {"seed": SEED, "random_task_count": RANDOM_TASK_COUNT,
            "task_count": len(variants), "same_task_order": True,
            "same_per_task_limits": True, "same_transition_draws": True,
            "feedback_arm": "source_returns_retained_and_refreshed",
            "matched_arm": "source_returns_deleted_after_each_visit",
            "frozen_screened_offers": True,
            "fixed_per_task_allocation": True,
            "allocation_policy": "common_minimum_resources_per_task",
            "external_model_calls": 0},
        "action_divergence_positions": divergences, "arms": arms}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    summary = {"status": payload["status"], "tasks": len(variants),
               "action_divergences": divergences,
               "feedback_costs": arms["feedback"]["total_costs"],
               "matched_costs": arms["matched_delete"]["total_costs"],
               "output": str((OUT / "results.json").relative_to(ROOT))}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
