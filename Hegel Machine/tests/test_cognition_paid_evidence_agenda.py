"""A fresh agreement cannot erase a conflict in the paid response history."""

from copy import deepcopy
from dataclasses import replace
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


@pytest.fixture(scope="module")
def recorded():
    path = Path(__file__).parent / "fixtures/paid_evidence_agenda_root07.json.gz"
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def laboratory_factory():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(scripts))
        return importlib.import_module("run_observed_condition_decoding_study").laboratory


def restored_report(payload):
    report = json.loads(json.dumps(payload))
    for batch in report["batches"]:
        history = []
        for row in batch["observations"]:
            row["history"] = list(history)
            history.append({**row["features"], "__target__": row["target"]})
    return report


def paid_rows(laboratory, report):
    return [row for batch in paid_view(laboratory, report, report["selected_channels"])
            for row in batch]


@pytest.fixture(scope="module")
def normal_run(recorded, laboratory_factory):
    config = replace(ResearchConfig(**recorded["config"]), seed=recorded["task"]["seed"])
    library = KnowledgeLibrary.from_dict(deepcopy(recorded["before_task_library"]))
    with laboratory_factory(recorded["task"]) as laboratory:
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=recorded["measurement_budget"]).run(laboratory)
        rows = paid_rows(laboratory, report)
        return report, rows, dict(laboratory.costs)


def test_normal_run_repairs_actual_paid_conflict_before_new_independent_confirmation(
        recorded, laboratory_factory, normal_run):
    original = restored_report(recorded["conflicting_report"])
    with laboratory_factory(recorded["task"]) as laboratory:
        old_paid = paid_rows(laboratory, original)
    old_program = Program.from_dict(original["program"])
    assert original["confirmation_loss"] < 1e-7
    assert loss(old_program, old_paid, original["fallback"], .5) == pytest.approx(.4251497086503138)
    assert original["condition_uncertainty"]["conditions"][0]["needs_threshold_revision"]
    assert original["condition_uncertainty"]["conditions"][0]["conditional_loss"] < 1e-7

    report, rows, lab_costs = normal_run
    actions = report["actions"]
    repairs = [(index, event) for index, event in enumerate(actions)
               if event["action"] == "revise_condition_on_paid_response" and event["changed_program"]]
    assert repairs
    first_repair_index, first_repair = repairs[0]
    assert first_repair["before_loss"] > 1e-7 > first_repair["after_loss"]
    assert first_repair["evidence_rows"] > 24 and first_repair["measured_rows"] == 0
    assert first_repair["requires_fresh_confirmation"] and first_repair["compute_cost"] > 0
    for event in actions:
        if (event["action"] == "choose_observed_response_followup"
                and event["paid_response_loss"] > 1e-7):
            assert event["research_action"] in ("revise_paid_condition", "reconstruct_observed_response")
            assert event["evidence_basis"] == "all_paid_original_response"

    program = Program.from_dict(report["program"])
    assert program != old_program
    # All earlier purchased counterexamples still support the returned model,
    # including the points absent from the old lucky confirmation trajectories.
    assert loss(program, old_paid, report["fallback"], .5) < 1e-7
    assert loss(program, rows, report["fallback"], .5) < 1e-7
    assert report["condition_uncertainty"]["original_target_loss"] < 1e-7
    assert not any(item["needs_threshold_revision"]
                   for item in report["condition_uncertainty"]["conditions"])
    assert report["confirmation_loss"] is not None and report["confirmation_loss"] < 1e-7
    last_repair_index = repairs[-1][0]
    confirmations = [(index, event) for index, event in enumerate(actions)
        if index > last_repair_index and event["action"] == "validate_frozen_program"
        and event["accepted"] and event["program"] == report["program"]]
    assert confirmations
    confirmation_index, confirmation = confirmations[-1]
    acquire_index, acquisition = next((index, event) for index, event in reversed(
        list(enumerate(actions[:confirmation_index]))) if event["action"] == "acquire"
        and event["purpose"] == "confirm_frozen_program")
    assert acquire_index > last_repair_index
    assert acquisition["purpose"] == "confirm_frozen_program"
    assert acquisition["probe_id"] not in {event["probe_id"] for event in actions[:last_repair_index]
                                           if event["action"] == "acquire"}
    batch = next(batch for batch in report["batches"]
                 if batch["probe"]["probe_id"] == acquisition["probe_id"])
    assert acquisition["observation_cost"] == acquisition["measurement_cost"] == len(batch["observations"])
    assert confirmation["observation_rows"] == len(batch["observations"])
    assert loss(program, [Observation(**item) for item in batch["observations"]], report["fallback"], .5) < 1e-7
    for resource in ("observation_cost", "measurement_cost"):
        assert report["costs"][resource] == lab_costs[resource]
        assert report["costs"][resource] == sum(event[resource] for event in actions if event["action"] == "acquire")
    assert report["costs"]["compute_cost"] <= recorded["config"]["compute_budget"]
    assert first_repair_index < acquire_index


def test_json_continuation_with_no_fresh_rows_keeps_the_question_unresolved(recorded, laboratory_factory):
    previous = restored_report(recorded["conflicting_report"])
    frozen = deepcopy(previous)
    old_costs = previous["costs"]
    config = replace(ResearchConfig(**recorded["config"]), seed=recorded["task"]["seed"],
        observation_budget=old_costs["observation_cost"], compute_budget=old_costs["compute_cost"] + 12000)
    library = KnowledgeLibrary.from_dict(deepcopy(recorded["after_task_library"]))
    with laboratory_factory(recorded["task"]) as laboratory:
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=old_costs["measurement_cost"]).run(laboratory, previous=previous)
        rows = paid_rows(laboratory, report)
    events = report["actions"][len(previous["actions"]):]
    assert report["status"] == "unresolved"
    assert not any(event["action"] == "acquire" for event in events)
    assert report["batches"] == previous["batches"] and previous == frozen
    for resource in ("observation_cost", "measurement_cost", "queries"):
        assert report["costs"][resource] == old_costs[resource]
        assert report["visit_costs"][resource] == 0
    assert 0 <= report["visit_costs"]["compute_cost"] <= 12000
    if report["program"] != previous["program"]:
        assert loss(Program.from_dict(report["program"]), rows, report["fallback"], .5) < 1e-7
        assert report["confirmation_loss"] is None
        assert report["research_program"]["status"] == "requires_fresh_validation"
    else:
        assert report["condition_uncertainty"]["status"] != "supported"


def test_same_paid_evidence_retains_the_better_explanation_without_false_conflict(
        recorded, laboratory_factory):
    previous = restored_report(recorded["supported_report"])
    frozen = deepcopy(previous)
    task = recorded["supported_task"]
    wrong = recorded["conflicting_report"]
    candidate = {"program": wrong["program"], "channels": wrong["selected_channels"],
        "kind": wrong["selected_kind"], "fallback": wrong["fallback"],
        "research_program": wrong["research_program"], "condition_uncertainty": wrong["condition_uncertainty"]}
    old_costs = previous["costs"]
    config = replace(ResearchConfig(**recorded["config"]), seed=task["seed"],
        observation_budget=old_costs["observation_cost"], compute_budget=old_costs["compute_cost"] + 12000)
    library = KnowledgeLibrary.from_dict(deepcopy(recorded["after_task_library"]))
    with laboratory_factory(task) as laboratory:
        rows = paid_rows(laboratory, previous)
        old_loss = loss(Program.from_dict(previous["program"]), rows, previous["fallback"], .5)
        wrong_loss = loss(Program.from_dict(candidate["program"]), rows, candidate["fallback"], .5)
        assert old_loss < 1e-7 < wrong_loss
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=old_costs["measurement_cost"]).run(laboratory,
                previous=previous, continuation_candidate=candidate)
    events = report["actions"][len(previous["actions"]):]
    assert report["program"] == previous["program"]
    assert report["batches"] == previous["batches"] and previous == frozen
    assert not any(event["action"] == "revise_condition_on_paid_response" for event in events)
    assert not any(event["action"] == "choose_observed_response_followup"
        and event["research_action"] in ("revise_paid_condition", "reconstruct_observed_response") for event in events)
    assert not any(event["action"] == "acquire" for event in events)
    for resource in ("observation_cost", "measurement_cost", "queries"):
        assert report["costs"][resource] == old_costs[resource]
