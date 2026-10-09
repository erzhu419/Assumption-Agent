#!/usr/bin/env python3
"""Attribute a measured source return with source-specific retain/delete arms."""

from copy import deepcopy
from dataclasses import replace
from math import isfinite
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "source_feedback_attribution_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from run_source_feedback_task_family import ParametricConflictLab, _fixtures, _offers, _previous


SEED = 20261005
VARIANTS = (
    {"label": "base_0", "current_weight": .55, "lag_weight": 1.2, "noise": 0.},
    {"label": "shifted", "current_weight": .6, "lag_weight": 1.2, "noise": 0.},
    {"label": "base_1", "current_weight": .55, "lag_weight": 1.2, "noise": 0.},
    {"label": "noisy", "current_weight": .55, "lag_weight": 1.2, "noise": .02},
)


class SessionLab(ParametricConflictLab):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _config(knowledge_tie_tolerance=0.0):
    return ResearchConfig(observation_budget=100, compute_budget=20_000,
        minimum_improvement=1e-7, max_rounds=1, active_probes=False,
        risk_learning=False, seed=SEED, max_candidates=2,
        knowledge_tie_tolerance=knowledge_tie_tolerance)


def _earn_positive_return(case, good, library):
    """Earn the good source return from one real continuation before replay.

    The four arms below are then clones of this paid state.  This keeps the
    attribution contrast about retaining/deleting an observed return rather
    than about inventing a score in the replay fixture.
    """
    from hegel_machine.cognition.models import SearchBudget
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_revisits import offer_successor_knowledge

    previous = deepcopy(case["previous"])
    previous["task_id"] = "positive_source_attribution"
    lab = type(case["lab"])(case["spec"])
    agent = UnifiedResearchAgent(_config(), library=library, measurement_budget=0)
    offer = offer_successor_knowledge(agent, lab, previous, {good}, SearchBudget(4_000))
    if offer is None:
        raise RuntimeError("real positive source continuation was not offered")
    report = agent.run(lab, previous=previous,
                       continuation_candidate=offer["candidate"])
    positive = next((row for row in report["knowledge_return"]["returns"]
                     if row["source_id"] == good and row["retained"]
                     and row["paired_gain"] > 0), None)
    if positive is None:
        raise RuntimeError("positive source continuation did not produce a retained gain")
    return {key: positive[key] for key in (
        "source_id", "condition", "paired_gain", "retained_normalized_gain",
        "adaptation_compute", "comparison_compute", "confirmation_status")
            if key in positive}


def _prepare(case, good, library, variants=None):
    variants = VARIANTS if variants is None else variants
    tasks, specs, pending = [], [], []
    for position, variant in enumerate(variants):
        spec = replace(case["spec"], task_id=f"attribution_{variant['label']}")
        lab = ParametricConflictLab(spec, current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"], seed=SEED + position)
        previous = _previous(case, lab)
        config, offers = _offers(case, good, library, lab, previous, SEED + position)
        for offer in offers:
            offer.update(task_position=position, source_task_position=position)
        tasks.append({"task_index": position, "seed": SEED + position,
                      "variant": variant["label"],
                      "research_limits": {"observation_budget": config.observation_budget,
                          "compute_budget": config.compute_budget, "measurement_budget": 0}})
        specs.append((spec, variant))
        pending.extend(offers)
    return tasks, specs, pending


def _delete_returns(library, mode, bad, good):
    deleted = {"delete_bad": {bad}, "delete_good": {good}, "delete_all": {bad, good}}.get(mode, set())
    for concept in library.concepts:
        concept.reuse_history = [item for item in concept.reuse_history
            if item.get("kind") != "research_return" or item.get("source_id") not in deleted]


def _counterfactual_source_replays(case, warm, prepared):
    """Measure each frozen source continuation on the same task response.

    These replays are an attribution readout, not extra scheduler budget: each
    source gets a fresh library clone and the same paid prefix.  Keeping this
    separate from the four mask arms prevents a source's intrinsic quality
    from being mistaken for a causal scheduler effect.
    """
    rows = []
    for position, (spec, variant) in enumerate(prepared[1]):
        for offer in (item for item in prepared[2] if item["task_position"] == position):
            from hegel_machine.cognition.evaluation import loss
            from hegel_machine.cognition.language import Program

            lab = ParametricConflictLab(spec, current_weight=variant["current_weight"],
                lag_weight=variant["lag_weight"], noise=variant["noise"],
                seed=SEED + position)
            previous = _previous(case, lab)
            library = KnowledgeLibrary.from_dict(deepcopy(warm.to_dict()))
            agent = __import__(
                "hegel_machine.cognition.unified_research",
                fromlist=["UnifiedResearchAgent"],
            ).UnifiedResearchAgent(_config(), library=library, measurement_budget=0)
            report = agent.run(lab, previous=previous,
                               continuation_candidate=deepcopy(offer["candidate"]))
            heldout_lab = ParametricConflictLab(spec,
                current_weight=variant["current_weight"], lag_weight=variant["lag_weight"],
                noise=variant["noise"], seed=SEED + position + 100_000)
            heldout_loss = loss(Program.from_dict(report["program"]),
                                heldout_lab.observe(spec.available_probes[-1]),
                                report["fallback"], .5)
            if not isfinite(heldout_loss):
                raise RuntimeError(f"non-finite counterfactual loss at {position}")
            rows.append({
                "task_position": position,
                "variant": variant["label"],
                "source_concept_ids": list(offer["candidate"]["source_concept_ids"]),
                "status": report["status"],
                "heldout_loss": heldout_loss,
                "visit_costs": dict(report["visit_costs"]),
                "external_model_calls": 0,
            })
    return rows


def _tie_controlled_good_return(case, good, warm):
    """Isolate the good-return tie-break after removing the bad return.

    Production selection must let feedback break only an equal immediate-gain
    tie.  The natural fixture has a small intrinsic gain difference, so this
    controlled replay records that estimand explicitly instead of calling the
    natural four-arm null an attribution result.
    """
    tie_library = KnowledgeLibrary.from_dict(deepcopy(warm.to_dict()))
    _delete_returns(tie_library, "delete_bad", case["source_id"], good)
    prepared = _prepare(case, good, tie_library)
    pending = prepared[2]
    for position in range(len(VARIANTS)):
        offers = [item for item in pending if item["task_position"] == position]
        tie_gain = min(item["expected_loss_reduction"] for item in offers)
        for offer in offers:
            offer["expected_loss_reduction"] = tie_gain
            offer["relative_gain"] = tie_gain
    limits = {}
    for position in range(len(VARIANTS)):
        offers = [item for item in pending if item["task_position"] == position]
        limits[position] = {
            key: max(item["minimum_resources"][key] for item in offers)
            for key in ("observation_cost", "measurement_cost", "compute_cost")}
    retained = _run_arm(case, good, tie_library, "retain_all", prepared, limits)
    deleted = _run_arm(case, good, tie_library, "delete_good", prepared, limits)
    retained["arm"] = "retain_good_tie"
    deleted["arm"] = "delete_good_tie"
    retained_rows, deleted_rows = retained["tasks"], deleted["tasks"]
    return {"protocol": {
        "removed_before_arms": [case["source_id"]],
        "equalized_field": "expected_loss_reduction_and_relative_gain",
        "same_prepared_offers": True,
        "fixed_execution_limits": limits,
    }, "arms": {"retain_good_tie": retained, "delete_good_tie": deleted},
        "contrast": {
            "action_divergence_count": sum(
                a["selected_source"] != b["selected_source"]
                for a, b in zip(retained_rows, deleted_rows)),
            "heldout_loss_delta": [
                (a["heldout_loss"] - b["heldout_loss"])
                if a["heldout_loss"] is not None and b["heldout_loss"] is not None else None
                for a, b in zip(retained_rows, deleted_rows)],
        }}


def _run_arm(case, good, warm, mode, prepared, allocation_limits,
             knowledge_tie_tolerance=0.0):
    config = _config(knowledge_tie_tolerance)
    library = KnowledgeLibrary.from_dict(deepcopy(warm.to_dict()))
    bad = case["source_id"]
    _delete_returns(library, mode, bad, good)
    tasks, specs, pending = prepared

    def factory(item):
        spec, variant = specs[item["task_index"]]
        return SessionLab(spec, current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"], seed=item["seed"])

    session = UnifiedResearchSession(config, tasks=tasks, library=library,
        measurement_budget=0, lab_factory=factory)
    session.state["results"] = [{"task_position": position, "research": deepcopy(
        _previous(case, ParametricConflictLab(spec, current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"], seed=SEED + position)))
        } for position, (spec, variant) in enumerate(specs)]
    session.state["pending_revisits"] = deepcopy(pending)
    session.state["screening_compute_cost"] = 0
    rows = []
    for position, (spec, variant) in enumerate(specs):
        session._refresh_source_feedback()
        candidates = [item for item in session.state["pending_revisits"]
                      if item["task_position"] == position]
        if not candidates:
            rows.append({"task_position": position, "variant": variant["label"], "arm": mode,
                "transition": {key: variant[key] for key in ("current_weight", "lag_weight", "noise")},
                "selected_source": [], "status": "withdrawn_before_execution",
                "heldout_loss": None, "visit_costs": {"observation_cost": 0,
                    "measurement_cost": 0, "compute_cost": 0, "queries": 0},
                "allocation_limits": dict(allocation_limits[position]),
                "scheduler_decision": None, "feedback_events": deepcopy(session.state["scheduler_events"]),
                "external_model_calls": 0})
            continue
        from hegel_machine.cognition.models import SearchBudget
        from hegel_machine.cognition.research_portfolio import choose_research_visit
        remaining = session._remaining()
        search_budget = SearchBudget(remaining["compute_cost"])
        choice = choose_research_visit(candidates, remaining, search_budget,
            allocation_for=lambda offer, _others, _remaining: dict(offer["minimum_resources"]),
            knowledge_tie_tolerance=config.knowledge_tie_tolerance)
        session.state["screening_compute_cost"] += search_budget.spent
        chosen = choice["offer"] if choice is not None else None
        before = session._costs()
        executed = bool(chosen is not None and session._perform_revisit(
            chosen, allocation_limits=allocation_limits[position]))
        if not executed:
            rows.append({"task_position": position, "variant": variant["label"], "arm": mode,
                "transition": {key: variant[key] for key in ("current_weight", "lag_weight", "noise")},
                "selected_source": [], "status": "not_selected",
                "heldout_loss": None, "visit_costs": {"observation_cost": 0,
                    "measurement_cost": 0, "compute_cost": 0, "queries": 0},
                "scheduler_decision": choice["decision"] if choice else None,
                "scheduler_search_compute": search_budget.spent,
                "allocation_limits": dict(allocation_limits[position]),
                "cost_delta": {key: session._costs()[key] - before[key] for key in before},
                "feedback_events": deepcopy(session.state["scheduler_events"]),
                "external_model_calls": 0})
            session.state["pending_revisits"] = [item for item in session.state["pending_revisits"]
                if item["task_position"] != position]
            _delete_returns(session.library, mode, bad, good)
            continue
        report = session.state["results"][position]["research"]
        after = session._costs()
        heldout_lab = ParametricConflictLab(spec, current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"], seed=SEED + position + 100_000)
        heldout_loss = loss(Program.from_dict(report["program"]),
                            heldout_lab.observe(spec.available_probes[-1]), report["fallback"], .5)
        if not isfinite(heldout_loss):
            raise RuntimeError(f"non-finite loss at {position}/{mode}")
        rows.append({"task_position": position, "variant": variant["label"], "arm": mode,
            "transition": {key: variant[key] for key in ("current_weight", "lag_weight", "noise")},
            "selected_source": list(chosen["candidate"]["source_concept_ids"]),
            "status": report["status"], "heldout_loss": heldout_loss,
            "visit_costs": dict(report["visit_costs"]),
            "cumulative_costs": {key: after[key] for key in after},
            "cost_delta": {key: after[key] - before[key] for key in before},
            "scheduler_decision": choice["decision"],
            "scheduler_search_compute": search_budget.spent,
            "allocation_limits": dict(allocation_limits[position]),
            "feedback_events": deepcopy(session.state["scheduler_events"]),
            "return_records": {concept.concept_id: sum(item.get("kind") == "research_return"
                for item in concept.reuse_history) for concept in session.library.concepts},
            "external_model_calls": 0})
        session.state["pending_revisits"] = [item for item in session.state["pending_revisits"]
            if item["task_position"] > position]
        _delete_returns(session.library, mode, bad, good)
    return {"arm": mode, "deleted_source_ids": sorted(
        {bad, good} if mode == "delete_all" else {bad} if mode == "delete_bad" else
        {good} if mode == "delete_good" else set()), "tasks": rows,
        "total_costs": session._costs(), "pending_after": len(session.state["pending_revisits"]),
        "external_model_calls": 0}


def main():
    case, good, warm, _cold = _fixtures()
    modes = ("retain_all", "delete_bad", "delete_good", "delete_all")
    positive_return = _earn_positive_return(case, good, warm)
    prepared = _prepare(case, good, warm)
    allocation_limits = {}
    for position in range(len(VARIANTS)):
        offers = [item for item in prepared[2] if item["task_position"] == position]
        if not offers:
            raise RuntimeError(f"no prepared offers at task {position}")
        allocation_limits[position] = {
            key: max(item["minimum_resources"][key] for item in offers)
            for key in ("observation_cost", "measurement_cost", "compute_cost")}
    arms = {mode: _run_arm(case, good, warm, mode, prepared, allocation_limits)
            for mode in modes}
    counterfactuals = _counterfactual_source_replays(case, warm, prepared)
    tie_controlled = _tie_controlled_good_return(case, good, warm)
    mask_contrasts = {}
    for left, right in (("retain_all", "delete_good"),
                        ("retain_all", "delete_bad"),
                        ("delete_bad", "delete_all")):
        left_rows, right_rows = arms[left]["tasks"], arms[right]["tasks"]
        mask_contrasts[f"{left}_vs_{right}"] = {
            "action_divergence_count": sum(
                a["selected_source"] != b["selected_source"]
                for a, b in zip(left_rows, right_rows)),
            "heldout_loss_delta": [
                (a["heldout_loss"] - b["heldout_loss"])
                if a["heldout_loss"] is not None and b["heldout_loss"] is not None else None
                for a, b in zip(left_rows, right_rows)],
        }
    assert positive_return.get("confirmation_status") == "confirmed"
    assert tie_controlled["contrast"]["action_divergence_count"] == 2
    tie_retained = tie_controlled["arms"]["retain_good_tie"]["tasks"]
    tie_deleted = tie_controlled["arms"]["delete_good_tie"]["tasks"]
    assert tie_retained[0]["selected_source"] == [good]
    assert tie_retained[2]["selected_source"] == [good]
    assert tie_deleted[0]["selected_source"] == [case["source_id"]]
    assert tie_deleted[2]["selected_source"] == [case["source_id"]]
    assert all(len(arm["tasks"]) == len(VARIANTS) for arm in arms.values())
    assert all(row["external_model_calls"] == 0 for arm in arms.values() for row in arm["tasks"])
    payload = {"status": "complete", "lineage": {
        "package_version": "0.181.0", "parent": "source-feedback random retention",
        "stage": "source-specific attribution and withdrawal"},
        "protocol": {"seed": SEED, "task_count": len(VARIANTS),
            "same_task_order": True, "same_per_task_limits": True,
            "same_prepared_offers": True, "fixed_execution_limits": allocation_limits,
            "arms": list(modes), "external_model_calls": 0},
        "positive_return_acquisition": positive_return,
        "counterfactual_source_replays": counterfactuals,
        "tie_controlled_good_return": tie_controlled,
        "mask_contrasts": mask_contrasts,
        "source_ids": {"bad": case["source_id"], "good": good}, "arms": arms}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": payload["status"], "arms": list(modes),
        "output": str((OUT / "results.json").relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
