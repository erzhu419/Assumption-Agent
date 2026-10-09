#!/usr/bin/env python3
"""Shared-budget multi-task sequence for source-feedback action choices."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "source_feedback_long_horizon_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from run_source_feedback_task_family import ParametricConflictLab, _fixtures, _offers, _previous


VARIANTS = (
    {"label": "base_law", "current_weight": .55, "lag_weight": 1.2, "noise": 0.},
    {"label": "shifted_current_law", "current_weight": .6, "lag_weight": 1.2, "noise": 0.},
    {"label": "noisy_base_law", "current_weight": .55, "lag_weight": 1.2, "noise": .02},
)
TASK_VARIANTS = VARIANTS * 2
SEED = 1237


class SessionLab(ParametricConflictLab):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _config():
    return ResearchConfig(observation_budget=100, compute_budget=20_000,
        minimum_improvement=1e-7, max_rounds=1, active_probes=False,
        risk_learning=False, seed=SEED, max_candidates=2)


def _prepare(case, good, library):
    tasks, specs, pending = [], [], []
    for position, variant in enumerate(TASK_VARIANTS):
        spec = replace(case["spec"], task_id=f"long_{variant['label']}_{position}")
        lab = ParametricConflictLab(spec, current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"], seed=SEED + position)
        previous = _previous(case, lab)
        config, offers = _offers(case, good, library, lab, previous, SEED + position)
        for offer in offers:
            offer.update(task_position=position, source_task_position=position)
        tasks.append({"task_index": position, "seed": SEED + position,
                      "variant": variant["label"],
                      "research_limits": {"observation_budget": config.observation_budget,
                          "compute_budget": config.compute_budget,
                          "measurement_budget": 0}})
        specs.append((spec, variant))
        pending.extend(offers)
    return tasks, specs, pending


def _run_arm(case, good, base_library, arm):
    config = _config()
    library = KnowledgeLibrary.from_dict(deepcopy(base_library.to_dict()))
    tasks, specs, pending = _prepare(case, good, library)
    if arm == "matched_no_return":
        payload = library.to_dict()
        for concept in payload["concepts"]:
            concept["reuse_history"] = [item for item in concept.get("reuse_history", [])
                if item.get("kind") != "research_return"]
        library = KnowledgeLibrary.from_dict(payload)

    def factory(item):
        spec, variant = specs[item["task_index"]]
        return SessionLab(spec, current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"], seed=item["seed"])

    session = UnifiedResearchSession(config, tasks=tasks, library=library,
        measurement_budget=0, lab_factory=factory)
    session.state["results"] = [{"task_position": position,
        "research": deepcopy(previous)} for position, previous in enumerate(
            [_previous(case, ParametricConflictLab(spec,
                current_weight=variant["current_weight"], lag_weight=variant["lag_weight"],
                noise=variant["noise"], seed=SEED + position))
             for position, (spec, variant) in enumerate(specs)])]
    session.state["pending_revisits"] = deepcopy(pending)
    session.state["screening_compute_cost"] = 0
    rows = []
    for position, (spec, variant) in enumerate(specs):
        session._refresh_source_feedback()
        candidates = [item for item in session.state["pending_revisits"]
                      if item["task_position"] == position]
        if not candidates:
            rows.append({"task_position": position, "variant": variant["label"],
                         "status": "withdrawn_before_execution", "arm": arm})
            continue
        chosen = candidates[0]
        before = session._costs()
        executed = session._perform_revisit(chosen)
        if not executed:
            raise RuntimeError(f"shared-budget revisit failed at task {position}/{arm}")
        report = session.state["results"][position]["research"]
        after = session._costs()
        heldout_lab = ParametricConflictLab(spec, current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"], seed=SEED + position + 100_000)
        heldout = heldout_lab.observe(spec.available_probes[-1])
        rows.append({"task_position": position, "variant": variant["label"], "arm": arm,
            "selected_source": list(chosen["candidate"]["source_concept_ids"]),
            "status": report["status"], "heldout_loss": loss(
                Program.from_dict(report["program"]), heldout, report["fallback"], .5),
            "visit_costs": dict(report["visit_costs"]),
            "cumulative_costs": {key: after[key] for key in after},
            "cost_delta": {key: after[key] - before[key] for key in after},
            "feedback_events": deepcopy(session.state["scheduler_events"]),
            "external_model_calls": 0})
        # One continuation per task is the frozen long-horizon protocol. Drop
        # the rival and any dynamically generated quote for the visited task.
        session.state["pending_revisits"] = [item for item in session.state["pending_revisits"]
            if item["task_position"] > position]
        if arm == "matched_no_return":
            for concept in session.library.concepts:
                concept.reuse_history = [item for item in concept.reuse_history
                    if item.get("kind") != "research_return"]
    costs = session._costs()
    return {"arm": arm, "tasks": rows, "total_costs": costs,
        "pending_after": len(session.state["pending_revisits"]),
        "external_model_calls": 0}


def main():
    case, good, warm, cold = _fixtures()
    arms = {"feedback": _run_arm(case, good, warm, "feedback"),
            "matched_no_return": _run_arm(case, good, cold, "matched_no_return")}
    for arm in arms.values():
        assert len(arm["tasks"]) == len(TASK_VARIANTS)
        assert arm["pending_after"] == 0
        assert arm["external_model_calls"] == 0
    feedback = arms["feedback"]["tasks"]
    matched = arms["matched_no_return"]["tasks"]
    for index, variant in enumerate(TASK_VARIANTS):
        assert feedback[index]["variant"] == matched[index]["variant"] == variant["label"]
        assert feedback[index]["visit_costs"] == matched[index]["visit_costs"]
    for index, variant in enumerate(TASK_VARIANTS):
        if variant["label"] == "base_law":
            assert feedback[index]["selected_source"] == [good]
        else:
            assert feedback[index]["selected_source"] == [case["source_id"]]
        assert matched[index]["selected_source"] == [case["source_id"]]
    assert arms["feedback"]["total_costs"] == arms["matched_no_return"]["total_costs"]
    payload = {"status": "complete", "protocol": {
        "task_count": len(TASK_VARIANTS), "same_task_order": True,
        "same_per_task_limits": True, "shared_total_budget": True,
        "feedback_arm": "source_returns_retained_and_refreshed",
        "matched_arm": "source_returns_removed_after_each_visit",
        "external_model_calls": 0}, "arms": arms}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": payload["status"], "tasks": len(TASK_VARIANTS),
                      "output": str((OUT / "results.json").relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
