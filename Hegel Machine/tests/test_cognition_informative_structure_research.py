"""Actual imperfect duration evidence remains a fallible complete-law nominee."""

from copy import deepcopy
from dataclasses import replace
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.observed_partial_structure import (
    partial_structure_competitors, partial_structure_nominations)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


@pytest.fixture(scope="module")
def recorded():
    with gzip.open(Path(__file__).parent / "fixtures/informative_structure_root03.json.gz", "rt") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def factory():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return importlib.import_module("run_observed_condition_decoding_study").laboratory


def rows_from(batch):
    history, rows = [], []
    for item in batch["observations"]:
        rows.append(Observation(dict(item["features"]), item["target"], tuple(history)))
        history.append({**item["features"], "__target__": item["target"]})
    return rows


def agent_before(recorded):
    return UnifiedResearchAgent(replace(ResearchConfig(**recorded["config"]), seed=recorded["task"]["seed"]),
        library=KnowledgeLibrary.from_dict(deepcopy(recorded["before_task_library"])),
        measurement_budget=recorded["measurement_budget"])


def test_measured_weak_gate_and_smooth_alternative_keep_actual_failures_after_json(recorded):
    saved = deepcopy(recorded)
    events = recorded["initial_paid_events"]
    nominations = partial_structure_nominations(events)
    assert {item["width"] for item in nominations} == {7}
    assert {item.get("placement") for item in nominations} == {None, "inside_window", "outside_window"}
    outside = next(item for item in nominations if item.get("placement") == "outside_window")
    assert outside["gain"] > 1e-7 and outside["loss"] > 1e-7
    assert not outside["response_supported"] and not outside["condition_identified"]
    for nominee in nominations:
        raw = next(item for item in events[1]["competing_structures"]
                   if (item["control"], item["width"], item.get("placement"))
                   == (nominee["control"], nominee["width"], nominee.get("placement")))
        assert all(nominee[key] == raw[key] for key in (
            "loss", "gain", "status", "response_supported", "condition_identified"))
        support = nominee["partial_support"]
        assert len(support) == 1 and support[0]["local_loss"] == raw["loss"]
        assert support[0]["background"] == events[1]["fixed_control_background"]
        assert support[0]["support_scope"] == "fixed_control_background"
    assert partial_structure_nominations(json.loads(json.dumps(events))) == nominations
    assert recorded == saved


def test_informative_duration_refits_complete_rivals_on_common_paid_responses(recorded, factory):
    saved = deepcopy(recorded)
    batches = recorded["paid_batches"]
    training = [row for index in (0, 1, 2, 4, 5) for row in rows_from(batches[index])]
    validation = rows_from(batches[3])
    assert len(training) == 84 and len(validation) == 24
    budget = SearchBudget(20000)
    agent = agent_before(recorded)
    with factory(recorded["task"]) as laboratory:
        spec = agent._spec(laboratory, ("meter",))
        claims = partial_structure_competitors(spec, training, recorded["initial_paid_events"], budget)
        validation_budget = SearchBudget(24 * len(claims))
        checked = agent._assess(claims, validation, validation_budget,
                                initial_prediction(training, .5), .5)
        assert laboratory.costs["observation_cost"] == laboratory.costs["measurement_cost"] == 0
    assert len(checked) == len(claims) == 3 and 0 < budget.spent <= 20000
    assert validation_budget.spent == 24 * len(claims)
    assert {claim["research_program"]["selection_steps"][0].get("placement") for claim in checked} == {
        None, "inside_window", "outside_window"}
    for claim in checked:
        assert claim["fit_rows"] == len(training)
        assert claim["research_program"]["status"] == "requires_fresh_validation"
        assert claim["training_loss"] == pytest.approx(loss(claim["candidate"].program,
            training, initial_prediction(training, .5), .5))
        assert claim["validation_loss"] == pytest.approx(loss(claim["candidate"].program,
            validation, initial_prediction(training, .5), .5))
        assert claim["partial_support"][0]["support_scope"] == "fixed_control_background"
        if claim["research_program"]["selection_steps"][0].get("placement") != "inside_window":
            assert not claim["partial_support"][0]["response_supported"]
    assert recorded == saved
    print(json.dumps({"informative_paid_fit": {"training_rows": len(training),
        "unfitted_validation_rows": len(validation), "fit_compute": budget.spent,
        "validation_compute": validation_budget.spent, "rivals": [{
            "placement": claim["research_program"]["selection_steps"][0].get("placement"),
            "training_loss": claim["training_loss"], "validation_loss": claim["validation_loss"]}
            for claim in checked]}}))


def test_normal_research_consumes_paid_temporal_clues_and_confirms_on_new_rows(recorded, factory, tmp_path):
    agent = agent_before(recorded)
    with factory(recorded["task"]) as laboratory:
        report = agent.run(laboratory)
        physical_costs = dict(laboratory.costs)
    restored = json.loads(json.dumps(report))
    actions = restored["actions"]
    durations = [event for event in actions if event["action"] == "distinguish_paid_condition_duration"]
    decisions = [event for event in actions if event["action"] == "choose_condition_research_investment"]
    boundary_plans = [event for event in actions if event["action"] == "plan_competing_observed_conditions"]
    confirmations = [event for event in actions if event["action"] == "validate_frozen_program"]
    trace = {"status": report["status"], "costs": report["costs"],
        "confirmation_loss": report["confirmation_loss"],
        "condition_status": (report.get("condition_uncertainty") or {}).get("status"),
        "condition_ambiguity": (report.get("condition_uncertainty") or {}).get("expected_prediction_ambiguity"),
        "decisions": [{key: event.get(key) for key in ("buy", "reason", "probability_of_new_information",
            "calibrated_information_probability", "resource_share", "fixed_control_background", "threshold_intervals",
            "experiment_revision_basis", "planning_compute_cost")}
            for event in decisions],
        "boundary_purchases": [{**{key: event.get(key) for key in ("probe_id", "fixed_control_background",
            "threshold_intervals", "experiment_revision_basis", "forecast_length", "boundary_disagreement")},
            "gap_hits": sum(interval["threshold_interval"][0] < row["features"][interval["condition_field"]]
                < interval["threshold_interval"][1] for interval in event["threshold_intervals"]
                for batch in report["batches"] if batch["probe"]["probe_id"] == event["probe_id"]
                for row in batch["observations"])} for event in boundary_plans],
        "confirmation_losses": [event["loss"] for event in confirmations]}
    (tmp_path / "development_informative_structure_trace.json").write_text(json.dumps(trace, indent=2))
    print(json.dumps({"normal_informative_structure_research": trace}))
    assert durations and decisions and confirmations
    nominations = partial_structure_nominations(actions)
    assert nominations == partial_structure_nominations(restored["actions"])
    assert any(item.get("gain", 0.) > 1e-7 and not item["response_supported"]
               for item in nominations)
    for nominee in nominations:
        raw = next(item for item in actions[nominee["observed_experiment_index"]]["competing_structures"]
                   if (item["control"], item["width"], item.get("placement"))
                   == (nominee["control"], nominee["width"], nominee.get("placement")))
        assert all(nominee[key] == raw[key] for key in ("loss", "gain", "status", "response_supported", "condition_identified"))
    first_background = durations[0]["fixed_control_background"]
    assert any(event["fixed_control_background"] != first_background for event in decisions)
    assert all(event["planning_compute_cost"] > 0 for event in decisions)
    boundary_decisions = [event for event in decisions
                          if event["experiment_revision_basis"] == "unmeasured_temporal_boundary"]
    assert boundary_decisions
    for event in boundary_plans:
        purchase = next(item for item in actions if item["action"] == "acquire" and item["probe_id"] == event["probe_id"])
        batch = next(item for item in report["batches"] if item["probe"]["probe_id"] == event["probe_id"])
        assert purchase["observation_cost"] == len(batch["observations"]) > 0
        assert all(point == event["fixed_control_background"] for point in batch["probe"]["sequence"])
        if event["experiment_revision_basis"] == "unmeasured_temporal_boundary":
            assert event["forecast_length"] == len(batch["observations"])
            assert event["boundary_disagreement"] > 0
            assert any(item["response_disagreement"] > 0 for item in event["interval_exposure"])
    for event in decisions:
        assert 0 <= event["calibrated_information_probability"] <= 1
        if not event["buy"] and event["reason"] == "retain_question_for_more_valuable_future_evidence":
            assert event["calibrated_information_probability"] < event["resource_share"]
    first_duration = next(index for index, event in enumerate(actions) if event["action"] == "distinguish_paid_condition_duration")
    fresh = [(index, event) for index, event in enumerate(actions)
             if index > first_duration and event["action"] == "validate_frozen_program"]
    assert fresh
    for index, confirmation in fresh:
        purchase_index, purchase = next((i, item) for i, item in reversed(list(enumerate(actions[:index])))
            if item["action"] == "acquire" and item["purpose"] == "confirm_frozen_program")
        assert purchase_index > first_duration
        batch = next(item for item in report["batches"] if item["probe"]["probe_id"] == purchase["probe_id"])
        assert confirmation["observation_rows"] == purchase["observation_cost"] == len(batch["observations"])
        assert confirmation["loss"] == pytest.approx(loss(Program.from_dict(confirmation["program"]),
            [Observation(**row) for row in batch["observations"]], report["fallback"], .5))
    if report["status"] == "completed":
        assert report["confirmation_loss"] is not None and report["confirmation_loss"] <= 1e-7
        assert (report.get("condition_uncertainty") or {}).get("status", "supported") == "supported"
    else:
        assert report["status"] == "unresolved"
    if not any(event["buy"] for event in boundary_decisions):
        assert report["status"] == "unresolved"
    for resource in ("observation_cost", "measurement_cost"):
        assert report["costs"][resource] == physical_costs[resource]
        assert physical_costs[resource] == sum(event[resource] for event in actions if event["action"] == "acquire")
    assert report["costs"]["observation_cost"] <= recorded["config"]["observation_budget"]
    assert report["costs"]["measurement_cost"] <= recorded["measurement_budget"]
    assert report["costs"]["compute_cost"] <= recorded["config"]["compute_budget"]
