"""Retain a paid local structure as a fallible full-response competitor."""

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
    with gzip.open(Path(__file__).parent / "fixtures/paid_structure_response_root05.json.gz", "rt") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def laboratory_factory():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
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
        print(json.dumps({"paid_structure_response": {
            "status": report["status"], "costs": report["costs"],
            "full_fits": [{"rows": event["fitting_rows"],
                "compute": event["compute_cost"],
                "structures": event["fitted_structures"]}
                for event in report["actions"]
                if event["action"] == "refit_observed_condition_response"],
            "frozen_confirmations": [{"rows": event["observation_rows"],
                "loss": event["loss"]} for event in report["actions"]
                if event["action"] == "validate_frozen_program"]}}, sort_keys=True))
        return report, paid_rows(laboratory, report), dict(laboratory.costs)


def test_normal_run_keeps_identified_inside_and_unresolved_outside_in_one_paid_fit(
        recorded, normal_run):
    report, rows, lab_costs = normal_run
    actions = report["actions"]
    local_index, local = next((index, event) for index, event in enumerate(actions)
        if event["action"] == "distinguish_paid_condition_duration"
        and any(item.get("width") == 7 and item.get("condition_identified")
                and "inside_window" in item.get("placement_candidates", [])
                for item in event["evidence"]))
    assert local["measured_rows"] == 24
    inside = next(item for item in local["evidence"] if item["width"] == 7)
    assert inside["loss"] < 1e-7
    outside = next(item for item in local["competing_structures"]
                   if item["width"] == 7 and item["placement"] == "outside_window")
    assert not outside["condition_identified"] and outside["gain"] > 1e-7
    fits = [(index, event) for index, event in enumerate(actions)
            if event["action"] == "refit_observed_condition_response"
            and {item["placement"] for item in event["structure_nominations"]
                 if item["observed_experiment_index"] == local_index and item["width"] == 7}
                >= {"inside_window", "outside_window"}]
    assert fits
    fit_index, fit = fits[0]
    assert fit_index > local_index
    assert fit["fitting_rows"] > local["measured_rows"]
    assert fit["candidate_count"] > 0 and fit["compute_cost"] > 0
    assert fit["comparison_basis"] == "original_response"
    assert fit["fitting_priority"] in ("current_supported_structure", "paid_structure_competition")
    assert any(item["width"] == 7 and item["placement"] == "inside_window"
               for item in fit["fitted_structures"])
    nominated_inside = next(item for item in fit["structure_nominations"]
        if item["observed_experiment_index"] == local_index
        and item["width"] == 7 and item["placement"] == "inside_window")
    assert nominated_inside["condition_identified"]
    # Identified here means the recorded constant-background experiment, not
    # a fitted response law for all control backgrounds or a fresh confirmation.
    assert nominated_inside["fit_rows"] == 24
    confirmations = [(index, event) for index, event in enumerate(actions)
        if index > fit_index and event["action"] == "validate_frozen_program"]
    assert confirmations
    confirmation_index, confirmation = confirmations[0]
    acquire_index, acquisition = next((index, event) for index, event in reversed(
        list(enumerate(actions[:confirmation_index]))) if event["action"] == "acquire"
        and event["purpose"] == "confirm_frozen_program")
    assert acquire_index > fit_index
    assert acquisition["probe_id"] not in {event["probe_id"] for event in actions[:fit_index]
                                           if event["action"] == "acquire"}
    batch = next(batch for batch in report["batches"]
                 if batch["probe"]["probe_id"] == acquisition["probe_id"])
    assert acquisition["observation_cost"] == acquisition["measurement_cost"] == len(batch["observations"])
    assert confirmation["observation_rows"] == len(batch["observations"])
    measured = loss(Program.from_dict(confirmation["program"]),
                    [Observation(**row) for row in batch["observations"]], report["fallback"], .5)
    assert measured == pytest.approx(confirmation["loss"], abs=1e-7)
    for resource in ("observation_cost", "measurement_cost"):
        assert report["costs"][resource] == lab_costs[resource]
        assert report["costs"][resource] == sum(event[resource] for event in actions if event["action"] == "acquire")
    assert report["costs"]["compute_cost"] <= recorded["config"]["compute_budget"]
    if report["status"] == "completed":
        assert loss(Program.from_dict(report["program"]), rows, report["fallback"], .5) < 1e-7


def test_actual_local_support_and_paid_counterexamples_cannot_complete_an_unfunded_visit(
        recorded, laboratory_factory):
    previous = restored_report(recorded["negative_report"])
    frozen = deepcopy(previous)
    local = next(event for event in previous["actions"] if event["action"] == "distinguish_paid_condition_duration")
    assert local["response_supported"] and local["evidence"][0]["loss"] < 1e-7
    assert any(event["action"] == "confirm_observation_research"
               and event["after_loss"] > 1e-7 for event in previous["actions"])
    costs = previous["costs"]
    config = replace(ResearchConfig(**recorded["config"]), seed=recorded["task"]["seed"],
        observation_budget=costs["observation_cost"], compute_budget=costs["compute_cost"] + 12000)
    library = KnowledgeLibrary.from_dict(deepcopy(recorded["before_task_library"]))
    with laboratory_factory(recorded["task"]) as laboratory:
        rows = paid_rows(laboratory, previous)
        assert loss(Program.from_dict(previous["program"]), rows, previous["fallback"], .5) > 1e-7
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=costs["measurement_cost"]).run(laboratory, previous=previous)
    events = report["actions"][len(previous["actions"]):]
    assert report["status"] == "unresolved"
    assert report["batches"] == previous["batches"] and previous == frozen
    assert not any(event["action"] == "acquire" for event in events)
    for resource in ("observation_cost", "measurement_cost", "queries"):
        assert report["costs"][resource] == costs[resource]
        assert report["visit_costs"][resource] == 0
    if report["program"] != previous["program"]:
        assert report["confirmation_loss"] is None
        assert report["research_program"]["status"] == "requires_fresh_validation"


def test_common_paid_responses_keep_the_better_incumbent_against_the_real_failed_transfer(
        recorded, laboratory_factory):
    previous = restored_report(recorded["prior_report"])
    frozen = deepcopy(previous)
    task, wrong = recorded["prior_task"], recorded["negative_report"]
    candidate = {"program": wrong["program"], "channels": wrong["selected_channels"],
        "kind": wrong["selected_kind"], "fallback": wrong["fallback"],
        "research_program": wrong["research_program"], "condition_uncertainty": wrong["condition_uncertainty"]}
    costs = previous["costs"]
    config = replace(ResearchConfig(**recorded["config"]), seed=task["seed"],
        observation_budget=costs["observation_cost"], compute_budget=costs["compute_cost"] + 12000)
    library = KnowledgeLibrary.from_dict(deepcopy(recorded["before_task_library"]))
    with laboratory_factory(task) as laboratory:
        rows = paid_rows(laboratory, previous)
        assert loss(Program.from_dict(previous["program"]), rows, previous["fallback"], .5) < 1e-7
        assert loss(Program.from_dict(candidate["program"]), rows, candidate["fallback"], .5) > 1e-7
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=costs["measurement_cost"]).run(laboratory,
                previous=previous, continuation_candidate=candidate)
    assert report["program"] == previous["program"]
    assert report["batches"] == previous["batches"] and previous == frozen
    assert not any(event["action"] == "acquire" for event in report["actions"][len(previous["actions"]):])
    for resource in ("observation_cost", "measurement_cost", "queries"):
        assert report["costs"][resource] == costs[resource]
