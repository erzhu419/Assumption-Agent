"""Learned programs guide paid initial measurements without hiding later changes."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.measurement_research import MeasurementResearchAgent
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig


def _config():
    return ResearchConfig(observation_budget=96, compute_budget=60000,
                          max_rounds=3, seed=7)


def _restore(payload):
    return KnowledgeLibrary.from_dict(json.loads(payload))


def _fresh_loss(laboratory, result):
    rows = laboratory.evaluation_rows(result["selected_channels"], seed=3001,
                                     length=32, trajectories=2)
    return loss(Program.from_dict(result["episode"]["program"]), rows,
                result["episode"]["fallback"], .5)


@pytest.fixture(scope="module")
def learned_acquisition_models():
    library = KnowledgeLibrary()
    with make_lab(0, 18313) as laboratory:
        source = MeasurementResearchAgent(_config(), measurement_budget=384,
                                          library=library).run(laboratory)
        assert source["status"] == "completed" and _fresh_loss(laboratory, source) < 1e-8
    linear = json.dumps(library.to_dict())
    with make_lab(3, 48334) as laboratory:
        late = MeasurementResearchAgent(_config(), measurement_budget=384,
                                        library=library).run(laboratory)
        assert late["selected_channels"] == ["sensor_a", "sensor_b"]
        assert late["status"] == "completed" and _fresh_loss(laboratory, late) < 1e-6
    return {"linear": linear, "late": json.dumps(library.to_dict())}


def _paid_within_budget(laboratory, result):
    for kind, limit in (("observation", 96), ("measurement", 384)):
        key = f"{kind}_cost"
        assert result["costs"][key] == sum(query[key] for query in laboratory.acquisitions)
        assert result["costs"][key] <= limit
    assert result["costs"]["compute_cost"] <= _config().compute_budget


def test_restored_knowledge_buys_only_the_needed_initial_instrument(learned_acquisition_models):
    library = _restore(learned_acquisition_models["linear"])
    source_ids = {concept.concept_id for concept in library.concepts}
    with make_lab(0, 58341) as laboratory:
        agent = MeasurementResearchAgent(_config(), measurement_budget=384, library=library)
        initial = agent.run(laboratory, max_steps=1)
        first_queries = list(laboratory.acquisitions)
        assert len(first_queries) == 2
        assert all(query["channels"] == ["sensor_a"] for query in first_queries)
        assert initial["selected_channels"] == ["sensor_a"]
        incumbent = initial["research_state"]["workspace"]["incumbent"]
        assert incumbent["memory_id"] in source_ids
        assert incumbent["operation"] == "reuse" and incumbent["program"]["op"] == "call"
        assert all("sensor_b" not in row["features"]
                   for batch in initial["pilot_observations"] for row in batch)
        completed = agent.run(laboratory, research_state=initial["research_state"])
        assert completed["status"] == "completed" and _fresh_loss(laboratory, completed) < 1e-8
        _paid_within_budget(laboratory, completed)
    with make_lab(0, 58341) as laboratory:
        agent = MeasurementResearchAgent(_config(), measurement_budget=384)
        cold_initial = agent.run(laboratory, max_steps=1)
        cold = agent.run(laboratory, research_state=cold_initial["research_state"])
        assert _fresh_loss(laboratory, cold) < 1e-8
    assert initial["costs"]["measurement_cost"] < cold_initial["costs"]["measurement_cost"]
    assert completed["costs"]["measurement_cost"] < cold["costs"]["measurement_cost"]


def test_wrong_initial_model_recovers_on_the_paid_cheap_view(learned_acquisition_models):
    library = _restore(learned_acquisition_models["linear"])
    source, = library.concepts
    with make_lab(1, 28320) as laboratory:
        agent = MeasurementResearchAgent(_config(), measurement_budget=384, library=library)
        initial = agent.run(laboratory, max_steps=1)
        assert all(query["channels"] == ["sensor_a"] for query in laboratory.acquisitions[:2])
        completed = agent.run(laboratory, research_state=initial["research_state"])
        assert completed["status"] == "completed"
        assert completed["episode"]["selected_memory_id"] != source.concept_id
        assert _fresh_loss(laboratory, completed) < 1e-6
        fresh = laboratory.evaluation_rows(completed["selected_channels"], seed=3001,
                                          length=32, trajectories=2)
        assert loss(source.program, fresh, completed["episode"]["fallback"], .5) > .05
        _paid_within_budget(laboratory, completed)


@pytest.mark.parametrize("knowledge", ["linear", "late"])
def test_cheap_initial_agreement_does_not_hide_later_mechanism_activation(
        learned_acquisition_models, knowledge):
    library = _restore(learned_acquisition_models[knowledge])
    with make_lab(3, 58341) as laboratory:
        agent = MeasurementResearchAgent(_config(), measurement_budget=384, library=library)
        initial = agent.run(laboratory, max_steps=1)
        assert initial["selected_channels"] == ["sensor_a"]
        assert initial["research_state"]["view_validation_loss"] < 1e-8
        if knowledge == "linear":
            assert all(query["channels"] == ["sensor_a"] for query in laboratory.acquisitions)
        else:
            # Prior late-drive evidence may already justify measuring both
            # channels. Its dormant readings still cannot end the inquiry.
            assert all(row["features"].get("sensor_b", 0.) == 0.
                       for batch in initial["pilot_observations"] for row in batch)
        completed = agent.run(laboratory, research_state=initial["research_state"])
        assert completed["status"] == "completed"
        assert completed["selected_channels"] == ["sensor_a", "sensor_b"]
        assert any(event["adopted"] and event["evidence_source"] == "new_measurements"
                   for event in completed["measurement_events"])
        later_measurements = [batch for batch in completed["research_state"]["batches"]
                              if "sensor_b" in batch["channels"]]
        assert later_measurements
        assert any(abs(row["features"]["sensor_b"]) > 0
                   for batch in later_measurements for row in batch["observations"])
        assert _fresh_loss(laboratory, completed) < 1e-6
        _paid_within_budget(laboratory, completed)


def test_acquisition_plan_survives_json_and_process_restart_without_rebuying_pilots(
        learned_acquisition_models, tmp_path):
    library = _restore(learned_acquisition_models["late"])
    path = tmp_path / "acquisition.json"
    with make_lab(3, 78355, backend="process") as laboratory:
        agent = MeasurementResearchAgent(_config(), measurement_budget=384, library=library)
        initial = agent.run(laboratory, max_steps=1)
        path.write_text(json.dumps({"state": initial["research_state"],
                                    "knowledge": library.to_dict()}))
        uninterrupted = agent.run(laboratory, research_state=initial["research_state"])
        assert uninterrupted["status"] == "completed"
    saved = json.loads(path.read_text())
    with make_lab(3, 78355, backend="process") as laboratory:
        agent = MeasurementResearchAgent(_config(), measurement_budget=384,
            library=KnowledgeLibrary.from_dict(saved["knowledge"]))
        resumed = agent.run(laboratory, research_state=saved["state"])
        assert resumed["status"] == "completed"
        assert resumed["costs"] == uninterrupted["costs"]
        assert resumed["episode"]["program"] == uninterrupted["episode"]["program"]
        assert resumed["research_state"]["apparatus_acquisitions"] == (
            uninterrupted["research_state"]["apparatus_acquisitions"])
        assert sum(query["probe_id"] in {"instrument_pilot_0", "instrument_pilot_1"}
                   for query in laboratory.acquisitions) == 2
        assert _fresh_loss(laboratory, resumed) < 1e-6
        _paid_within_budget(laboratory, resumed)


def test_short_budget_waits_for_learned_coverage_then_records_success_after_resume(
        learned_acquisition_models, tmp_path):
    library = _restore(learned_acquisition_models["linear"])
    source_record, = library.measurements.to_dict()
    assert source_record["coverage_rows"] == 72
    with make_lab(0, 58341) as laboratory:
        limited = replace(_config(), observation_budget=64)
        stopped = MeasurementResearchAgent(limited, measurement_budget=384,
            library=library).run(laboratory)
        assert stopped["status"] == "budget_exhausted"
        assert stopped["research_state"]["stage"] == "coverage"
        assert stopped["episode"]["validation_loss"] < 1e-8
        question = stopped["measurement_question"]
        assert question["reason"] == "unreached_learned_coverage"
        assert question["status"] == "awaiting_budget"
        assert stopped["costs"]["observation_cost"] < question["required_rows"] == 72
        assert stopped["costs"]["observation_cost"] <= limited.observation_budget
        old_queries = list(laboratory.acquisitions)
    path = tmp_path / "unfinished_coverage.json"
    path.write_text(json.dumps({"state": stopped["research_state"],
                                "knowledge": library.to_dict()}))
    saved = json.loads(path.read_text())
    restored = KnowledgeLibrary.from_dict(saved["knowledge"])
    previous_records = len(restored.measurements.records)
    with make_lab(0, 58341, backend="process") as laboratory:
        completed = MeasurementResearchAgent(_config(), measurement_budget=384,
            library=restored).run(laboratory, research_state=saved["state"])
        assert completed["status"] == "completed"
        assert completed["measurement_question"].get("status") != "awaiting_budget"
        assert completed["costs"]["observation_cost"] >= question["required_rows"]
        assert laboratory.acquisitions[:len(old_queries)] == old_queries
        assert sum(query["probe_id"] in {"instrument_pilot_0", "instrument_pilot_1"}
                   for query in laboratory.acquisitions) == 2
        assert _fresh_loss(laboratory, completed) < 1e-8
        _paid_within_budget(laboratory, completed)
    assert len(restored.measurements.records) == previous_records + 1
    experience = restored.measurements.records[-1]
    assert experience["successful"] and experience["source_ids"]
    assert experience["coverage_rows"] >= source_record["coverage_rows"]
