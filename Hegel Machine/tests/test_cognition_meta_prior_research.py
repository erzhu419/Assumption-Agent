"""A public apparatus must drive hypothesis discovery and recursive revision."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import EvaluationContext, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.meta_prior_research import MetaPriorResearch


path = Path(__file__).resolve().parents[1] / "scripts" / "run_meta_prior_research_study.py"
module_spec = importlib.util.spec_from_file_location("meta_prior_study", path)
apparatus = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(apparatus)


def config(**changes):
    return replace(ResearchConfig(observation_budget=96, compute_budget=60000,
        minimum_improvement=1e-7, max_rounds=5, seed=173), **changes)


def evaluate(result, environment, seed=951):
    rows = apparatus.fresh_rows(environment, seed, count=128)
    program = Program.from_dict(result["program"])
    return loss(program, rows, result["fallback"], .5), program, rows


@pytest.fixture(scope="module")
def sequence_result():
    library, reports, knowledge = KnowledgeLibrary(), [], []
    for index, environment in enumerate(apparatus.sequence(31001)):
        report = MetaPriorResearch(config(seed=173+1009*index), library).run(environment)
        assert report["costs"]["observation_cost"] == environment.observation_count <= 96
        assert report["costs"]["compute_cost"] <= 60000
        reports.append(report)
        knowledge.append(deepcopy(library.to_dict()))
        library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    return reports, knowledge, library


def test_world_hypotheses_predict_new_responses_and_use_a_learned_coordinate(sequence_result):
    reports, _, _ = sequence_result
    results = [evaluate(report, environment) for report, environment in zip(reports, apparatus.sequence(31001))]
    assert all(error < 1e-6 for error, _, _ in results)
    _, program, rows = results[1]
    context = EvaluationContext(rows[0])
    program.predict(rows[0], context=context)
    assert context.call_evaluations > 0
    assert program.required_features() == {"x", "z"}
    assert results[3][1].required_features() == {"x"}


def test_real_counterexample_tests_an_adequate_replacement_before_buying_a_residual_fit(sequence_result):
    reports, _, _ = sequence_result
    before, after = reports[1], reports[2]
    environment = apparatus.sequence(31001)[2]
    rows = apparatus.fresh_rows(environment, 310019, count=128)
    old = Program.from_dict(before["program"])
    new = Program.from_dict(after["program"])
    assert loss(old, rows, before["fallback"], .5) > .05
    assert loss(new, rows, after["fallback"], .5) < 1e-6
    assert after["research_state"]["counterexamples"]
    assert not after["research_state"]["graph"].get("nodes")
    assert any(event["action"] == "defer_recursive_hypothesis" for event in after["events"])
    assert after["costs"]["observation_cost"] > len(after["research_state"]["queries"][0]["sequence"])


def test_revising_a_world_claim_preserves_its_successful_source_program(sequence_result):
    reports, snapshots, final_library = sequence_result
    original = KnowledgeLibrary.from_dict(snapshots[0])
    assert original.concepts and snapshots[0]["meta_priors"]
    final = {concept.concept_id: concept for concept in final_library.concepts}
    for concept in original.concepts:
        assert concept.concept_id in final
        assert final[concept.concept_id].program == concept.program
        for scope in concept.scopes:
            assert any(current.task_id == scope.task_id and current.support_examples == scope.support_examples
                       for current in final[concept.concept_id].scopes)
    error, _, _ = evaluate(reports[0], apparatus.sequence(31001)[0])
    assert error < 1e-6


def test_json_resume_retains_paid_observations_and_matches_uninterrupted_result():
    full_lab, paused_lab = apparatus.sequence(51001)[0], apparatus.sequence(51001)[0]
    full = MetaPriorResearch(config()).run(full_lab)
    library = KnowledgeLibrary()
    first = MetaPriorResearch(config(), library).run(paused_lab, max_steps=1)
    assert first["status"] != "completed"
    assert first["costs"]["observation_cost"] == paused_lab.observation_count > 0
    restored_library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    saved = json.loads(json.dumps(first["research_state"]))
    resumed = MetaPriorResearch(config(), restored_library).run(paused_lab, research_state=saved)
    assert resumed["costs"] == full["costs"]
    assert resumed["program"] == full["program"]
    assert resumed["costs"]["observation_cost"] == paused_lab.observation_count == full_lab.observation_count


def test_unfunded_validation_does_not_promote_an_unverified_world_claim():
    environment = apparatus.sequence(71001)[0]
    library = KnowledgeLibrary()
    prepared = MetaPriorResearch(config(), library).run(environment, max_steps=2)
    paid = prepared["costs"]["observation_cost"]
    assert prepared["research_state"]["candidates"] and not prepared["claims"]
    result = MetaPriorResearch(config(observation_budget=paid), library).run(environment,
        research_state=json.loads(json.dumps(prepared["research_state"])))
    assert result["costs"]["observation_cost"] == environment.observation_count == paid
    assert result["costs"]["compute_cost"] <= 60000
    assert not library.meta_priors
    assert result["status"] != "completed"


def test_nested_learned_response_is_retained_inside_the_executed_residual_answer():
    library, reports = KnowledgeLibrary(), []
    for index, environment in enumerate(apparatus.sequence(31001, "nested_readout")[:3]):
        report = MetaPriorResearch(config(seed=173+1009*index), library).run(environment)
        reports.append(report)
        assert report["costs"]["observation_cost"] == environment.observation_count <= 96
        assert report["costs"]["compute_cost"] <= 60000
        library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    # Evaluation begins after all three learning tasks have stopped. The final
    # task exceeds a raw quadratic model's expressive family, so merely building
    # an unused dependency graph cannot pass this functional comparison.
    fresh = apparatus.sequence(31001, "nested_readout")
    measured = [evaluate(report, environment, seed=1901+index)
                for index, (report, environment) in enumerate(zip(reports, fresh))]
    assert all(error < 1e-6 for error, _, _ in measured)
    final, program, rows = reports[2], measured[2][1], measured[2][2]
    previous = Program.from_dict(reports[1]["program"])
    assert loss(previous, rows, reports[1]["fallback"], .5) > .05
    nodes = final["research_state"]["graph"]["nodes"]
    selected = [node for node in nodes.values() if node.get("depends_on") and node.get("answer")
                and Program.from_dict(node["answer"]).expanded().simplified() == program.expanded().simplified()]
    assert selected
    assert any(nodes[identifier].get("residual_base") for node in selected for identifier in node["depends_on"])
    context = EvaluationContext(rows[0])
    program.predict(rows[0], context=context)
    called = {definition.source_concept_id for definition, _ in context._calls}
    assert reports[1]["prior_updates"][-1]["source_concept_id"] in called
