"""Partially identified paid structure survives and competes as a full law."""

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
    path = Path(__file__).parent / "fixtures/partial_structure_root08.json.gz"
    with gzip.open(path, "rt") as stream:
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


def paid_events(recorded):
    return [event for experiment in recorded["paid_experiments"]
            for event in (experiment["plan"], experiment["result"])]


def agent_before(recorded):
    return UnifiedResearchAgent(replace(ResearchConfig(**recorded["config"]),
        seed=recorded["task"]["seed"]),
        library=KnowledgeLibrary.from_dict(deepcopy(recorded["before_task_library"])),
        measurement_budget=recorded["measurement_budget"])


def test_zero_local_gain_keeps_width_and_equivalent_placements_after_json(recorded):
    saved = deepcopy(recorded)
    first = recorded["paid_experiments"][0]["result"]
    supported = [item for item in first["competing_structures"] if item["response_supported"]]
    assert {item["width"] for item in supported} == {7}
    assert all(abs(item["gain"]) < 1e-7 and not item["condition_identified"] for item in supported)
    nominations = partial_structure_nominations(paid_events(recorded))
    assert {item["width"] for item in nominations if item["response_supported"]} == {7}
    original_width = [item for item in nominations if item["width"] == 7]
    assert {item.get("placement") for item in original_width} == {None, "inside_window", "outside_window"}
    assert all(item["support_scope"] == "fixed_control_background"
               and not item["condition_identified"] for item in nominations)
    assert all(any(not support["response_supported"] and support["local_loss"] > 1e-7
                   for support in item["partial_support"]) for item in original_width)
    assert len({tuple(sorted(support["background"].items())) for item in original_width
                for support in item["partial_support"]}) == 3
    weaker = [item for item in nominations if item["width"] != 7]
    assert weaker
    for item in weaker:
        raw = next(other for other in paid_events(recorded)[item["observed_experiment_index"]]["competing_structures"]
                   if (other["control"], other["width"], other.get("placement"))
                   == (item["control"], item["width"], item.get("placement")))
        assert not item["response_supported"] and not item["condition_identified"]
        assert all(item[key] == raw[key] for key in ("loss", "gain", "status", "response_supported", "condition_identified"))
    assert partial_structure_nominations(json.loads(json.dumps(paid_events(recorded)))) == nominations
    assert recorded == saved


def test_three_paid_backgrounds_refit_common_complete_competitors(recorded, factory):
    saved = deepcopy(recorded)
    varied = recorded["initial_varied_batches"]
    training = [row for index in (0, 1, 2, 4) for row in rows_from(varied[index])]
    training.extend(row for experiment in recorded["paid_experiments"]
                    for row in rows_from(experiment["batch"]))
    validation = rows_from(varied[3])
    assert len(training) == 132 and len(validation) == 24
    budget, actions = SearchBudget(20000), []
    agent = agent_before(recorded)
    with factory(recorded["task"]) as laboratory:
        spec = agent._spec(laboratory, ("meter",))
        claims = partial_structure_competitors(spec, training, paid_events(recorded), budget)
        assert laboratory.costs["observation_cost"] == laboratory.costs["measurement_cost"] == 0
        checked = agent._assess(claims, validation, SearchBudget(24 * len(claims)),
                                initial_prediction(training, .5), .5)
    assert 0 < budget.spent <= 20000 and len(checked) == len(claims) >= 3
    assert {step.get("placement") for claim in claims
            for step in claim["research_program"]["selection_steps"]} == {
                None, "inside_window", "outside_window"}
    for claim in checked:
        assert claim["fit_rows"] == 132 and set(claim["fit_features"]) == set(spec.feature_names)
        assert claim["research_program"]["status"] == "requires_fresh_validation"
        assert claim["training_loss"] == pytest.approx(loss(claim["candidate"].program,
            training, initial_prediction(training, .5), .5))
        assert claim["validation_loss"] == pytest.approx(loss(claim["candidate"].program,
            validation, initial_prediction(training, .5), .5))
        assert any(not support["response_supported"] for support in claim["partial_support"])
    assert recorded == saved
    print(json.dumps({"shared_paid_structure_fit": {"training_rows": len(training),
        "unfitted_validation_rows": len(validation), "compute": budget.spent,
        "rivals": [{"placement": claim["research_program"]["selection_steps"][0].get("placement"),
            "training_loss": claim["training_loss"], "validation_loss": claim["validation_loss"]}
            for claim in checked]}}))


def test_normal_run_uses_partial_competition_for_a_new_background_and_paid_confirmation(
        recorded, factory, tmp_path):
    agent = agent_before(recorded)
    with factory(recorded["task"]) as laboratory:
        report = agent.run(laboratory)
        physical_costs = dict(laboratory.costs)
    restored = json.loads(json.dumps(report))
    actions = restored["actions"]
    fits = [event for event in actions if event["action"] == "fit_partial_structure_competitors"]
    plans = [event for event in actions if event["action"] == "plan_observed_condition_experiment"]
    confirmations = [event for event in actions if event["action"] == "validate_frozen_program"]
    trace = {"status": report["status"], "costs": report["costs"],
        "confirmation_loss": report["confirmation_loss"],
        "fits": [{key: event[key] for key in ("fit_rows", "fit_features", "compute_cost", "candidate_count")}
                 for event in fits],
        "plans": [{key: event.get(key) for key in ("probe_id", "fixed_control_background",
            "experiment_revision_basis", "partial_structure_disagreement", "forecast_length")}
            for event in plans],
        "confirmation_losses": [event["loss"] for event in confirmations]}
    (tmp_path / "development_partial_structure_trace.json").write_text(json.dumps(trace, indent=2))
    print(json.dumps({"normal_partial_structure_research": trace}))
    assert fits and all(event["compute_cost"] > 0 and event["fit_rows"] > 24 for event in fits)
    assert len({(event["fit_rows"], tuple(event["fit_features"])) for event in fits}) == len(fits)
    assert any(step.get("partial_support") for event in fits for claim in event["competitors"]
               for step in claim["research_program"]["selection_steps"])
    first_fit = next(index for index, event in enumerate(actions)
                     if event["action"] == "fit_partial_structure_competitors")
    updated = [event for index, event in enumerate(actions)
               if index > first_fit and event["action"] == "plan_observed_condition_experiment"]
    assert updated
    assert len({tuple(sorted(event["fixed_control_background"].items())) for event in plans}) >= 2
    for event in updated:
        if event["experiment_revision_basis"] == "cross_background_partial_support":
            assert event["partial_structure_disagreement"] > 0
            assert sum(item.get("partial_structure", False) for item in event["competing_hypotheses"]) >= 2
        else:
            # A complete fitted law can still fail the shared paid targets.
            # Its structural clue continues through real residual exploration,
            # without manufacturing value from those disproved parameters.
            assert event["experiment_revision_basis"] == "measured_background_misfit"
            assert event["expected_disagreement"] == 0.
            assert not event["competing_hypotheses"]
        purchase = next(item for item in actions if item["action"] == "acquire"
                        and item["probe_id"] == event["probe_id"])
        batch = next(item for item in report["batches"]
                     if item["probe"]["probe_id"] == event["probe_id"])
        assert purchase["observation_cost"] == len(batch["observations"]) > 0
        assert all(point == event["fixed_control_background"] for point in batch["probe"]["sequence"])
    assert partial_structure_nominations(restored["actions"]) == partial_structure_nominations(report["actions"])
    fresh = [(index, event) for index, event in enumerate(actions)
             if index > first_fit and event["action"] == "validate_frozen_program"]
    assert fresh
    for index, confirmation in fresh:
        purchase_index, purchase = next((i, item) for i, item in reversed(list(enumerate(actions[:index])))
            if item["action"] == "acquire" and item["purpose"] == "confirm_frozen_program")
        assert purchase_index > first_fit
        batch = next(item for item in report["batches"]
                     if item["probe"]["probe_id"] == purchase["probe_id"])
        assert confirmation["observation_rows"] == len(batch["observations"])
        assert confirmation["loss"] == pytest.approx(loss(Program.from_dict(confirmation["program"]),
            [Observation(**row) for row in batch["observations"]], report["fallback"], .5))
    if report["status"] == "completed":
        assert report["confirmation_loss"] <= recorded["config"]["minimum_improvement"]
        assert (report.get("condition_uncertainty") or {}).get("status", "supported") == "supported"
    elif report["confirmation_loss"] is not None and report["confirmation_loss"] > 1e-7:
        assert report["status"] == "unresolved"
    for resource in ("observation_cost", "measurement_cost"):
        assert report["costs"][resource] == physical_costs[resource]
        assert physical_costs[resource] == sum(event[resource] for event in actions if event["action"] == "acquire")
    assert report["costs"]["observation_cost"] <= recorded["config"]["observation_budget"]
    assert report["costs"]["measurement_cost"] <= recorded["measurement_budget"]
    assert report["costs"]["compute_cost"] <= recorded["config"]["compute_budget"]
