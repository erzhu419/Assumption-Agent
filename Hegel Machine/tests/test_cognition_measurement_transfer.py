"""Paid trajectories decide whether a learned instrument model earns reuse."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.measurement_research import MeasurementResearchAgent
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Probe, ResearchConfig


def _config():
    return ResearchConfig(observation_budget=96, compute_budget=60000,
                          max_rounds=3, seed=7)


def _fresh_loss(laboratory, result):
    rows = laboratory.evaluation_rows(result["selected_channels"], seed=3001,
                                     length=32, trajectories=2)
    program = Program.from_dict(result["episode"]["program"])
    return loss(program, rows, result["episode"]["fallback"], .5)


def _instrument(result):
    return next(item for item in result["selection"]
                if item["channels"] == ["sensor_a"])


@pytest.fixture(scope="module")
def learned_measurement():
    library = KnowledgeLibrary()
    with make_lab(0, 18313) as laboratory:
        source = MeasurementResearchAgent(_config(), measurement_budget=384,
                                         library=library).run(laboratory)
        assert source["selected_channels"] == ["sensor_a"]
        assert _fresh_loss(laboratory, source) < 1e-8
    assert len(library.concepts) == 1
    return json.dumps(library.to_dict()), library.concepts[0].concept_id


def test_restored_instrument_knowledge_saves_initial_search_and_is_actually_executed(
        learned_measurement, tmp_path):
    serialized, source_id = learned_measurement
    path = tmp_path / "measurement_knowledge.json"
    path.write_text(serialized)
    library = KnowledgeLibrary.from_dict(json.loads(path.read_text()))
    with make_lab(0, 58341, backend="process") as laboratory:
        agent = MeasurementResearchAgent(_config(), measurement_budget=384,
                                         library=library)
        initial = agent.run(laboratory, max_steps=1)
        selected = initial["research_state"]["workspace"]["incumbent"]
        assert initial["selected_channels"] == ["sensor_a"]
        assert selected["memory_id"] == source_id
        assert selected["operation"] == "reuse"
        assert selected["program"]["op"] == "call"
        reuse = _instrument(initial)["reuse"]
        assert reuse["skipped_search"] and reuse["selected_source"] == source_id
        trial, = reuse["trials"]
        assert trial["call"] and trial["source_concept_id"] == source_id
        assert max(trial["training_loss"], trial["validation_loss"]) < 1e-8
        assert _instrument(initial)["compute_cost"] == reuse["compute_cost"] == 48
        assert initial["costs"]["observation_cost"] == 48
        assert initial["costs"]["measurement_cost"] == 48
        assert initial["costs"]["compute_cost"] == (
            initial["initial_comparison_compute_cost"]
            + sum(item["compute_cost"] for item in initial["selection"]))
        completed = agent.run(laboratory,
            research_state=json.loads(json.dumps(initial["research_state"])))
        assert completed["status"] == "completed"
        assert _fresh_loss(laboratory, completed) < 1e-8
        assert len([query for query in laboratory.acquisitions
                    if query["probe_id"] in {"instrument_pilot_0", "instrument_pilot_1"}]) == 2
    with make_lab(0, 58341) as laboratory:
        cold = MeasurementResearchAgent(_config(), measurement_budget=384).run(laboratory)
        assert cold["selected_channels"] == completed["selected_channels"]
        assert _fresh_loss(laboratory, cold) < 1e-8
    assert _instrument(initial)["compute_cost"] < _instrument(cold)["compute_cost"]
    assert sum(item["compute_cost"] for item in initial["selection"]) < sum(
        item["compute_cost"] for item in cold["selection"])
    assert completed["episode"]["selected_memory_id"] == source_id
    reused = next(concept for concept in library.concepts if concept.concept_id == source_id)
    assert {"task_id": completed["task_id"], "kind": "executed_program"} in reused.reuse_history


def test_same_instrument_names_do_not_reuse_a_linear_model_for_a_nonlinear_response(
        learned_measurement):
    serialized, source_id = learned_measurement
    library = KnowledgeLibrary.from_dict(json.loads(serialized))
    source = next(concept for concept in library.concepts if concept.concept_id == source_id)
    with make_lab(1, 28320) as laboratory:
        agent = MeasurementResearchAgent(_config(), measurement_budget=384,
                                         library=library)
        initial = agent.run(laboratory, max_steps=1)
        selected = initial["research_state"]["workspace"]["incumbent"]
        assert initial["selected_channels"] == ["sensor_a"]
        assert selected["memory_id"] != source_id
        assert selected["operation"] != "reuse"
        reuse = _instrument(initial)["reuse"]
        assert not reuse["skipped_search"] and reuse["selected_source"] is None
        trial, = reuse["trials"]
        assert trial["source_concept_id"] == source_id
        assert min(trial["training_loss"], trial["validation_loss"]) > _config().minimum_improvement
        assert _instrument(initial)["compute_cost"] > reuse["compute_cost"] == 48
        completed = agent.run(laboratory, research_state=initial["research_state"])
        assert completed["status"] == "completed"
        assert _fresh_loss(laboratory, completed) < 1e-6
        fresh = laboratory.evaluation_rows(completed["selected_channels"], seed=3001,
                                          length=32, trajectories=2)
        assert loss(source.program, fresh, completed["episode"]["fallback"], .5) > .05
    assert completed["episode"]["selected_memory_id"] != source_id
    assert not any(event["kind"] == "executed_program"
                   and event["task_id"] == completed["task_id"] for event in source.reuse_history)


def test_memory_ablation_does_not_use_instrument_knowledge(learned_measurement):
    serialized, _ = learned_measurement
    config = replace(_config(), use_memory=False)
    library = KnowledgeLibrary.from_dict(json.loads(serialized))
    before = library.to_dict()
    results = []
    for knowledge in (library, KnowledgeLibrary()):
        with make_lab(0, 58341) as laboratory:
            results.append(MeasurementResearchAgent(config, measurement_budget=384,
                library=knowledge).run(laboratory, max_steps=1))
    retained, empty = results
    assert retained["selected_channels"] == empty["selected_channels"] == ["sensor_a"]
    assert retained["selection"] == empty["selection"]
    assert retained["costs"] == empty["costs"]
    assert retained["research_state"]["workspace"]["incumbent"]["memory_id"] is None
    assert library.to_dict() == before


def test_a_model_supported_by_only_one_pilot_does_not_skip_search(learned_measurement):
    serialized, source_id = learned_measurement
    library = KnowledgeLibrary.from_dict(json.loads(serialized))
    with make_lab(3, 58341) as laboratory:
        # This apparatus activates its second drive after 48 operating rows.
        # The two pilots straddle that real transition after prior operation.
        laboratory.acquire(Probe("prior_operation", tuple({"u": 0.} for _ in range(24))))
        agent = MeasurementResearchAgent(_config(), measurement_budget=384,
                                         library=library)
        initial = agent.run(laboratory, max_steps=1)
        instrument = _instrument(initial)
        trial, = instrument["reuse"]["trials"]
        assert trial["source_concept_id"] == source_id
        assert trial["training_loss"] < 1e-8
        assert trial["validation_loss"] > .05
        assert not instrument["reuse"]["skipped_search"]
        assert instrument["compute_cost"] > instrument["reuse"]["compute_cost"] == 48
        residual = agent.run(laboratory, research_state=initial["research_state"], max_steps=1)
        assert residual["research_state"]["stage"] == "reselect"
        acquired_before = len(laboratory.acquisitions)
        reselected = agent.run(laboratory, research_state=residual["research_state"], max_steps=1)
        event, = [item for item in reselected["measurement_events"]
                  if item["action"] == "recompare_measurements"]
        assert event["adopted"] and event["evidence_source"] == "new_measurements"
        reselect_probes = laboratory.acquisitions[acquired_before:]
        assert len(reselect_probes) == 2
        assert all(query["channels"] == ["sensor_a", "sensor_b"] for query in reselect_probes)
        queried = {probe["probe_id"] for probe in reselected["research_state"]["workspace"]["queried"]}
        assert {query["probe_id"] for query in reselect_probes} <= queried
        assert event["validation_probe_id"] == reselect_probes[1]["probe_id"]
        completed = agent.run(laboratory, research_state=reselected["research_state"])
        assert completed["status"] == "completed"
        assert completed["selected_channels"] == ["sensor_a", "sensor_b"]
        assert _fresh_loss(laboratory, completed) < 1e-6
        assert completed["costs"]["observation_cost"] <= 96
        assert completed["costs"]["measurement_cost"] <= 384
        assert completed["costs"]["compute_cost"] <= 60000
