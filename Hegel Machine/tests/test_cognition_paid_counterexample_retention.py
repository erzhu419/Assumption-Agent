"""Fresh ties preserve a response that repairs an already paid counterexample."""

from copy import deepcopy
from dataclasses import replace
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss, point_loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.numeric_method_revisits import record_numeric_method_visit
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def test_real_paid_counterexample_survives_a_tied_method_comparison_and_confirmation(monkeypatch):
    fixture = Path(__file__).parent / "fixtures/paid_counterexample_root11p2.json.gz"
    with gzip.open(fixture, "rt") as stream:
        recorded = json.load(stream)
    previous = deepcopy(recorded["report"])
    previous["batches"] = previous.pop("episodes")
    for batch in previous["batches"]:
        history = []
        for row in batch["observations"]:
            row["history"] = list(history)
            history.append({**row["features"], "__target__": row["target"]})
    offer = recorded["offer"]
    assert offer["task_position"] == 2 and offer["source_task_position"] == 4
    allocation = offer["minimum_resources"]
    config = replace(ResearchConfig(**recorded["config"]), seed=recorded["task"]["seed"],
        observation_budget=previous["costs"]["observation_cost"] + allocation["observation_cost"],
        compute_budget=previous["costs"]["compute_cost"] + allocation["compute_cost"])
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    laboratory = importlib.import_module("run_observed_condition_decoding_study").laboratory
    library = KnowledgeLibrary.from_dict(deepcopy(recorded["knowledge"]))
    with laboratory(recorded["task"]) as lab:
        paid = [row for batch in paid_view(lab, previous, previous["selected_channels"]) for row in batch]
        before = Program.from_dict(previous["program"])
        assert loss(before, paid, previous["fallback"], .5) == pytest.approx(.02154036533507375)
        source = next(batch for batch in previous["batches"]
            if batch["probe"]["probe_id"] == recorded["counterexample_probe_id"])
        item = source["observations"][recorded["counterexample_row_index"]]
        counterexample = Observation(item["features"], item["target"], tuple(item["history"]))
        assert point_loss(predict(before, counterexample, previous["fallback"]),
            counterexample.target, .5) == pytest.approx(5.945140832480354)
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=previous["costs"]["measurement_cost"] + allocation["measurement_cost"]).run(
                lab, previous=previous, continuation_candidate=offer["candidate"])
        record_numeric_method_visit(library, lab.spec_for(offer["candidate"]["channels"]),
            previous, report, offer["candidate"]["method_opportunity"], offer["screening_compute_cost"])
        events = report["actions"][len(previous["actions"]):]
        comparison = next(item for item in events if item["action"] == "compare_continuation_candidates")
        proposed = Program.from_dict(comparison["proposed_program"])
        assert proposed != before
        # This is the original fullfit's repair, not a replacement supplied by
        # the test. All earlier responses support it, including the failed point.
        assert loss(proposed, paid, previous["fallback"], .5) < 1e-18
        assert comparison["before_loss"] <= config.minimum_improvement
        assert comparison["after_loss"] <= config.minimum_improvement
        assert comparison["program"] == comparison["proposed_program"]
        assert report["program"] == comparison["proposed_program"]
        retained = Program.from_dict(report["program"])
        assert loss(retained, paid, report["fallback"], .5) < 1e-18
        assert point_loss(predict(retained, counterexample, report["fallback"]),
            counterexample.target, .5) < 1e-18
        fresh = report["continuation"]["method_visit_outcome"]["fresh_comparison"]
        assert fresh is not None
        assert fresh["before_loss"] <= config.minimum_improvement
        assert fresh["after_loss"] <= config.minimum_improvement
        acquisitions = [event for event in events if event["action"] == "acquire"]
        assert [event["purpose"] for event in acquisitions[:3]] == [
            "successor_partial_method_experiment", "compare_continuation_candidates",
            "confirm_continuation_program"]
        assert fresh["probe_id"] == acquisitions[2]["probe_id"]
        assert fresh["probe_id"] not in {event["probe_id"] for event in acquisitions[:2]}
        assert report["visit_costs"]["observation_cost"] == report["visit_costs"]["measurement_cost"] == 72
        assert report["visit_costs"]["compute_cost"] <= allocation["compute_cost"]
        assert report["batches"][:len(previous["batches"])] == previous["batches"]
