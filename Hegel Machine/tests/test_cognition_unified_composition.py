"""Learn across measurement scales, compose the result, then reuse its history."""

import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


@pytest.fixture(scope="module")
def composition_sequence():
    apparatus = {"environment": "objects", "apparatus_seed": 9013}
    tasks = [{**apparatus, "readout": "total", "seed": 10078},
             {**apparatus, "readout": "event", "seed": 20085},
             {**apparatus, "readout": "total_event", "seed": 30092},
             {**apparatus, "readout": "window_total_event", "seed": 50106},
             {**apparatus, "readout": "window_total_event", "seed": 70120,
              "rename_prefix": "later_"}]
    config = ResearchConfig(observation_budget=384, compute_budget=360000,
        minimum_improvement=1e-7, max_rounds=6, active_probes=True, seed=71)
    session = UnifiedResearchSession(config, tasks=tasks, measurement_budget=4608)
    source = session.run(max_tasks=3)
    assert source["status"] == "paused"
    assert source["evaluation"] == []
    saved = json.loads(json.dumps(source))
    resumed = UnifiedResearchSession(state=saved)
    completed = resumed.run()
    assert completed["results"][:3] == saved["results"]
    evaluated = resumed.evaluate(length=48, trajectories=3)
    assert evaluated["results"] == completed["results"]
    assert evaluated["costs"] == completed["costs"]
    return evaluated


def test_composed_research_predicts_independent_longer_sequences_and_renamed_measurements(composition_sequence):
    report = composition_sequence
    assert report["status"] == "completed"
    assert len(report["evaluation"]) == 5
    for result, score in zip(report["results"], report["evaluation"]):
        research = result["research"]
        assert score["loss"] < 1e-7, {
            "task": result["task_position"], "fresh_loss": score["loss"],
            "confirmation_loss": research["confirmation_loss"],
            "kind": research["selected_kind"], "costs": research["costs"]}
        assert research["costs"]["observation_cost"] <= 384
        assert research["costs"]["measurement_cost"] <= 4608
        assert research["costs"]["compute_cost"] <= 360000
    total, event, combined, window, renamed = [row["research"] for row in report["results"]]
    assert total["measurement_channels"] == ["coarse"]
    assert event["measurement_channels"] == ["fine"]
    assert all(not research["selected_channels"] for research in (total, event, combined, window, renamed))
    assert combined["costs"]["measurement_cost"] == window["costs"]["measurement_cost"] == 0
    assert renamed["costs"]["observation_cost"] == 24
    assert not any(name.startswith("measured_")
                   for name, _ in Program.from_dict(renamed["program"]).observed_variables())


def test_joint_and_temporal_research_save_executable_programs_and_actual_source_parents(composition_sequence):
    report = composition_sequence
    total, event, combined, window, renamed = [row["research"] for row in report["results"]]
    prerequisites = set(total["concepts_added"]) | set(event["concepts_added"])
    assert prerequisites <= set(_sources(Program.from_dict(combined["program"])))
    concepts = {item["concept_id"]: item for item in report["knowledge"]["concepts"]}
    priors = {item["source_concept_id"]: item for item in report["knowledge"]["meta_priors"]}
    for research in (combined, window):
        procedure = research["research_program"]
        assert procedure["status"] == "supported_on_fresh_trajectory"
        assert procedure["confirmation_rows"] > 0
        assert procedure["confirmation_loss"] < 1e-7
        assert procedure["source_ids"]
        assert len(procedure["selection_steps"]) >= 2
        nodes = procedure["graph"]["nodes"]
        assert nodes[procedure["output_node"]]["depends_on"]
        assert any(node["depends_on"] for name, node in nodes.items() if name.startswith("fit:"))
        for source_id in research["concepts_added"]:
            assert priors[source_id]["research_program"] == procedure
            assert set(procedure["source_ids"]) <= set(concepts[source_id]["parents"])
    assert set(combined["concepts_added"]) & set(_sources(Program.from_dict(window["program"])))
    assert set(window["concepts_added"]) & set(_sources(Program.from_dict(renamed["program"])))


def test_losing_retained_call_returns_its_already_paid_validation_failure(composition_sequence):
    report = composition_sequence
    combined, changed = [report["results"][index]["research"] for index in (2, 3)]
    source = combined["concepts_added"][0]
    failures = [event for event in changed["transfer"]
                if event.get("kind") == "failed_retained_call" and source in event["source_ids"]]
    assert failures
    training, validation = [[Observation(row["features"], row["target"], tuple(row["history"]))
                            for row in batch["observations"]] for batch in changed["batches"][:2]]
    event = failures[0]
    program = Program.from_dict(event["program"])
    assert program.op == "call"
    assert event["status"] == "counterexample" and not event["accepted"]
    assert event["validation_rows"] == len(validation)
    assert event["validation_loss"] == pytest.approx(
        loss(program, validation, initial_prediction(training, .5), .5))
    prior = next(item for item in report["knowledge"]["meta_priors"] if item["source_concept_id"] == source)
    assert prior["status"] == "conditional"
    assert any(item["kind"] == "failed_retained_call" and item["program"] == event["program"]
               for item in prior["counterexamples"])
    concept = next(item for item in report["knowledge"]["concepts"] if item["concept_id"] == source)
    assert concept["status"] != "revoked"


def test_conditional_source_keeps_earlier_coverage_when_original_response_returns(composition_sequence):
    report = composition_sequence
    source = report["results"][2]["research"]["concepts_added"][0]
    library = KnowledgeLibrary.from_dict(report["knowledge"])
    prior = next(item for item in library.meta_priors if item["source_concept_id"] == source)
    assert prior["status"] == "conditional" and prior["counterexamples"]
    prior_failures = len(prior["counterexamples"])
    returning = UnifiedResearchSession(ResearchConfig(**report["config"]),
        tasks=[{"environment": "objects", "apparatus_seed": 9013,
                "readout": "total_event", "seed": 80127}],
        library=library, measurement_budget=report["measurement_budget"])
    completed = returning.run()
    result = completed["results"][0]["research"]
    assert result["status"] == "completed"
    assert result["selected_kind"] == "retained_calibrated_call"
    assert source in _sources(Program.from_dict(result["program"]))
    assert result["costs"]["observation_cost"] == 24
    assert not result["concepts_added"]
    assert result["confirmation_loss"] < 1e-7
    assert returning.evaluate(length=48, trajectories=3)["evaluation"][0]["loss"] < 1e-7
    assert prior["status"] == "conditional"
    assert len(prior["counterexamples"]) >= prior_failures
