"""A method's measured rejection does not fund unchanged evidence repeatedly."""

from copy import deepcopy
from dataclasses import replace
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import point_loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.numeric_method_revisits import record_numeric_method_visit
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


ROOT = Path(__file__).resolve().parents[1]
BEFORE_VISIT = Path(__file__).parent / "fixtures/method_rejection_paid_root05.json.gz"


@pytest.fixture(scope="module")
def method_input():
    # Only this visit's original learning inputs are retained, without scoring.
    with gzip.open(BEFORE_VISIT, "rt") as stream:
        recorded = json.load(stream)
    report = recorded["report"]
    report["batches"] = report.pop("episodes")
    for batch in report["batches"]:
        history = []
        for row in batch["observations"]:
            row["history"] = list(history)
            history.append({**row["features"], "__target__": row["target"]})
    offer = recorded["offer"]
    assert offer["task_position"] == 2 and offer["source_task_position"] == 4
    return recorded


@pytest.fixture
def apparatus(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    study = importlib.import_module("run_observed_condition_decoding_study")
    return study.laboratory


def execute_method(recorded, factory):
    offer = deepcopy(recorded["offer"])
    previous = deepcopy(recorded["report"])
    allocation = offer["minimum_resources"]
    config = replace(ResearchConfig(**recorded["config"]), seed=recorded["task"]["seed"],
        observation_budget=previous["costs"]["observation_cost"] + allocation["observation_cost"],
        compute_budget=previous["costs"]["compute_cost"] + allocation["compute_cost"])
    library = KnowledgeLibrary.from_dict(deepcopy(recorded["knowledge"]))
    with factory(recorded["task"]) as lab:
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=previous["costs"]["measurement_cost"] + allocation["measurement_cost"]).run(
                lab, previous=previous, continuation_candidate=offer["candidate"])
        record_numeric_method_visit(library, lab.spec_for(offer["candidate"]["channels"]),
            previous, report, offer["candidate"]["method_opportunity"], offer["screening_compute_cost"])
    events = report["actions"][len(previous["actions"]):]
    decision = next(item for item in events if item["action"] == "choose_method_followup_research")
    return report, previous, decision, events, offer


def test_restored_process_confirms_the_actual_root05_response_repair(method_input, apparatus):
    # Since 0.164 continuation restores the apparatus random process. The old
    # root05 input now produces a valid repair rather than its earlier failed
    # replay. Unchanged-failure stopping is covered by method_joint_mismatch.
    report, previous, decision, events, offer = execute_method(method_input, apparatus)
    assert decision["continue"]
    assert decision["reason"] == "fresh_response_improved"
    assert decision["fresh_gain"] > method_input["config"]["minimum_improvement"]
    assert decision["new_max_point_loss"] <= decision["previous_max_point_loss"]
    assert decision["supported_structure"]
    assert report["program"] != previous["program"]
    assert report["status"] == "unresolved"
    fresh = report["continuation"]["method_visit_outcome"]["fresh_comparison"]
    assert fresh is not None and fresh["after_loss"] < method_input["config"]["minimum_improvement"]
    assert fresh["before_loss"] > method_input["config"]["minimum_improvement"]
    acquisitions = [item for item in events if item["action"] == "acquire"]
    assert [item["purpose"] for item in acquisitions] == [
        "successor_partial_method_experiment", "compare_continuation_candidates", "confirm_continuation_program"]
    assert report["visit_costs"]["observation_cost"] == 72
    assert report["visit_costs"]["compute_cost"] <= offer["minimum_resources"]["compute_cost"]
    lab = apparatus(method_input["task"])
    model = Program.from_dict(previous["program"])
    old_rows = [row for batch in paid_view(lab, previous, previous["selected_channels"]) for row in batch]
    old_peak = max(point_loss(predict(model, row, previous["fallback"]), row.target, .5) for row in old_rows)
    assert decision["previous_max_point_loss"] == pytest.approx(old_peak)
    assert decision["new_max_point_loss"] > fresh["before_loss"]


def test_new_actual_response_counterexample_reopens_method_research(method_input, apparatus):
    def changed_apparatus(task):
        original = type(apparatus(task))

        class ChangedResponse(original):
            def __init__(self):
                super().__init__()

            def acquire(self, probe, channels=()):
                batch = super().acquire(probe, channels)
                rows, history = [], []
                for index, row in enumerate(batch.observations):
                    target = row.target + (4. if index % 2 else -4.)
                    rows.append(Observation(dict(row.features), target, tuple(history)))
                    history.append({**row.features, "__target__": target})
                return MeasurementBatch(rows, batch.observation_cost, batch.measurement_cost, batch.channels)

        return ChangedResponse()

    funded = deepcopy(method_input)
    # The changed response justifies reconstruction, whose next independent
    # comparison and confirmation must also have actual purchased resources.
    funded["offer"]["minimum_resources"].update(observation_cost=168, measurement_cost=168)
    report, previous, decision, events, offer = execute_method(funded, changed_apparatus)
    assert decision["new_max_point_loss"] > decision["previous_max_point_loss"]
    assert decision["continue"] and decision["reason"] != "method_failed_without_new_information"
    assert report["continuation"]["method_visit_outcome"]["fresh_comparison"] is not None
    assert report["visit_costs"]["compute_cost"] <= offer["minimum_resources"]["compute_cost"]
    assert report["batches"][:len(previous["batches"])] == previous["batches"]
    continued = events[events.index(decision) + 1:]
    assert any(item["action"] in ("research_action_pending", "choose_condition_response_research",
        "replay_research_draft", "choose_risk_research") for item in continued)
