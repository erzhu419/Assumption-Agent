"""Structural explanations must work inside the shared, resumable researcher."""

import json
from dataclasses import replace

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession


CONFIG = ResearchConfig(observation_budget=384, compute_budget=360000,
    minimum_improvement=1e-7, max_rounds=6, active_probes=True, seed=71)


def test_large_measured_coordinate_model_can_be_simplified_saved_and_executed():
    # Object coordinates with quadratic/history terms exceed the old recursive
    # additive-chain limit, before any comparison could reject that model.
    terms = [Program("feature", name=f"coordinate_{index}") for index in range(512)]
    coefficients = [(-1 if index % 3 else 1) * (1 + index / 512) for index in range(512)]
    features = {term.name: (index % 11 - 5) / 7 for index, term in enumerate(terms)}
    program = _sum_terms(terms, coefficients, .375).simplified()
    restored = Program.from_dict(json.loads(json.dumps(program.to_dict())))
    expected = .375 + sum(coefficient * features[term.name]
                          for term, coefficient in zip(terms, coefficients))
    assert restored.predict(Observation(features, 0.)) == pytest.approx(expected, abs=1e-10)


def operators(program):
    expanded = program.expanded()
    return {expanded.op} | set().union(*(operators(child) for child in expanded.children))


def test_scalar_and_renamed_object_research_share_a_resumable_knowledge_library():
    tasks = [{"task_index": 0, "seed": 71},
             {"environment": "objects", "readout": "total", "seed": 71},
             {"environment": "objects", "readout": "total", "seed": 89,
              "rename_prefix": "meter_"}]
    session = UnifiedResearchSession(CONFIG, tasks=tasks, measurement_budget=4608)
    first = session.run(max_tasks=1)
    assert first["status"] == "paused"
    assert first["pending_tasks"] == [1, 2]
    assert first["evaluation"] == []
    saved = json.loads(json.dumps(first))
    resumed = UnifiedResearchSession(state=saved)
    completed = resumed.run()
    assert completed["results"][0] == saved["results"][0]
    assert completed["tasks"] == saved["tasks"]
    assert completed["status"] == "completed"
    source, renamed = [item["research"] for item in completed["results"][1:]]
    assert source["selected_channels"] == renamed["selected_channels"] == ["coarse"]
    assert renamed["selected_kind"] == "short_callable_transfer"
    assert set(source["concepts_added"]) & set(_sources(Program.from_dict(renamed["program"])))
    assert set(renamed["used_sources"]) <= set(completed["results"][2]["available_concepts"])
    assert all(name.startswith("measured_meter_")
               for name, _ in Program.from_dict(renamed["program"]).observed_variables())
    # Coarse selection must cause an actual cheaper acquisition after its
    # projection was compared using the already purchased fine readings.
    assert any(batch["channels"] == ["fine"] for batch in source["batches"])
    assert source["batches"][-1]["channels"] == ["coarse"]
    scored = resumed.evaluate(length=32, trajectories=2)
    assert scored["results"] == completed["results"]
    assert scored["costs"] == completed["costs"]
    assert all(score["loss"] < 1e-7 for score in scored["evaluation"])
    assert [score["measurement_cost"] for score in scored["evaluation"]] == [0, 192, 192]


@pytest.mark.parametrize("readout,required", [
    ("focus", "neighbor_sum"), ("event", "event_delta"), ("window", "window_sum")])
def test_structures_predict_fresh_sequences_without_mistaking_residuals_for_noise(readout, required):
    lab = UnifiedObjectLab(readout=readout, seed=71)
    report = UnifiedResearchAgent(replace(CONFIG, seed=89), measurement_budget=4608).run(lab)
    program = Program.from_dict(report["program"])
    rows = lab.evaluation_rows(report["selected_channels"], seed=815, length=32, trajectories=2)
    assert report["status"] == "completed"
    assert report["selected_channels"] == ["fine"]
    assert required in operators(program)
    assert loss(program, rows, report["fallback"], .5) < 1e-7
    assert report["noise_assessment"] is None
    assert report["costs"]["observation_cost"] <= CONFIG.observation_budget
    assert report["costs"]["measurement_cost"] <= 4608
    assert report["costs"]["compute_cost"] <= CONFIG.compute_budget
    assert any(action["action"] == "revise_structure_boundary" for action in report["actions"])


def test_renamed_successor_reopens_structure_after_old_interaction_boundary_fails():
    library = KnowledgeLibrary()
    source_lab = UnifiedObjectLab(readout="focus", seed=71)
    source = UnifiedResearchAgent(CONFIG, measurement_budget=4608, library=library).run(source_lab)
    method_uses = {prior["source_concept_id"]: prior.get("method_uses", 0)
                   for prior in library.meta_priors}
    successor_lab = UnifiedObjectLab(readout="focus", seed=113, rename_prefix="next_")
    successor = UnifiedResearchAgent(replace(CONFIG, seed=113), measurement_budget=4608,
                                     library=library).run(successor_lab)
    sources = set(source["concepts_added"])
    assert sources
    assert any(sources & set(event["source_ids"]) and event["status"] == "counterexample"
               for event in successor["transfer"])
    assert not sources & set(successor["used_sources"])
    assert any(prior["counterexamples"] for prior in library.meta_priors
               if prior["source_concept_id"] in sources)
    program = Program.from_dict(successor["program"])
    assert not _sources(program)
    method_sources = set(successor["method_source_ids"])
    assert method_sources & sources
    learned = [concept for concept in library.concepts
               if concept.concept_id in successor["concepts_added"]]
    assert learned and all(method_sources <= set(concept.parents) for concept in learned)
    for prior in library.meta_priors:
        if prior["source_concept_id"] in method_sources:
            assert prior["method_uses"] == method_uses[prior["source_concept_id"]] + 1
    assert "neighbor_sum" in operators(program)
    fresh = successor_lab.evaluation_rows(successor["selected_channels"], seed=815,
                                         length=32, trajectories=2)
    assert successor["status"] == "completed"
    assert loss(program, fresh, successor["fallback"], .5) < 1e-7


def test_event_adoption_uses_varied_sequences_instead_of_two_zero_response_setups():
    # This formal-cohort task previously accepted a wrong event law because
    # confirmation repeated only two endpoint setups, both with zero response.
    seed = 2949419113 + 10007 * 4
    lab = UnifiedObjectLab(readout="event", seed=seed)
    report = UnifiedResearchAgent(replace(CONFIG, seed=seed, use_memory=False),
                                  measurement_budget=4608).run(lab)
    confirmation_ids = {action["probe_id"] for action in report["actions"]
                        if action["action"] == "acquire"
                        and action["purpose"] == "confirm_frozen_program"}
    confirmations = [batch for batch in report["batches"]
                     if batch["probe"]["probe_id"] in confirmation_ids]
    assert confirmations
    for batch in confirmations:
        assert len({tuple(sorted(point.items())) for point in batch["probe"]["sequence"]}) > 2
        assert any(abs(row["target"]) > 1e-6 for row in batch["observations"])
    assert report["status"] == "completed", {
        key: report[key] for key in ("selected_kind", "confirmation_loss", "noise_assessment", "costs")}
    rows = lab.evaluation_rows(report["selected_channels"], seed=seed + 1000003,
                               length=64, trajectories=4)
    measured = loss(Program.from_dict(report["program"]), rows, report["fallback"], .5)
    assert measured < 1e-7, {"loss": measured, "selected_kind": report["selected_kind"]}
