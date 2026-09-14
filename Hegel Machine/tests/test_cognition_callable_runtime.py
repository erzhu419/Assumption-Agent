"""Learned functions must participate in later search, including after restore."""

import importlib.util
from pathlib import Path
import sys

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.memory import KnowledgeLibrary


spec = importlib.util.spec_from_file_location(
    "_callable_concept_demo", Path(__file__).resolve().parents[1] / "scripts" / "run_callable_concept_demo.py")
demo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = demo
spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def learned():
    return demo.learn_three_tasks()


def test_a_computation_learned_from_observations_becomes_an_atom_for_new_program_search(learned):
    source = learned["fits"]["A"]
    assert not demo.call_structure(source.candidate.program)["contains_call"]
    assert loss(source.candidate.program, learned["fresh"]["A"], 0., .5) < 1e-10
    assert learned["atoms"]["B"] and all(atom.op == "call" for atom in learned["atoms"]["B"])
    composed, raw = learned["fits"]["B_callable"], learned["fits"]["B_raw"]
    assert composed.budget == raw.budget
    for fitted in (composed, raw):
        assert 0 < fitted.search_cost + fitted.selection_cost <= fitted.budget
    assert demo.call_structure(composed.candidate.program)["contains_call"]
    assert loss(composed.candidate.program, learned["fresh"]["B_callable"], 0., .5) < 1e-8
    assert loss(raw.candidate.program, learned["fresh"]["B_raw"], 0., .5) > .01


def test_checkpoint_preserves_nested_calls_for_role_changed_tasks(learned):
    source_ids = learned["source_ids"]
    restored = KnowledgeLibrary.from_dict(learned["checkpoint"])
    b = next(concept for concept in restored.concepts if concept.concept_id == source_ids["B"])
    assert source_ids["A"] in b.parents
    c = learned["fits"]["C"].candidate.program
    structure = demo.call_structure(c)
    assert structure["maximum_call_depth"] >= 2
    assert {source_ids["A"], source_ids["B"]} <= set(structure["source_concept_ids"])
    assert c.required_features() == {"u", "v"}
    assert loss(c, learned["fresh"]["C"], 0., .5) < 1e-7
    deleted = learned["library"].without({source_ids["A"]})
    assert not {source_ids["A"], source_ids["B"], source_ids["C"]}.intersection(
        concept.concept_id for concept in deleted.concepts)


@pytest.mark.parametrize("task", ["B_callable", "C"])
def test_executable_calls_match_expansion_and_reuse_actual_computation(learned, task):
    program = learned["fits"][task].candidate.program
    measured = demo.execution_comparison(program, learned["fresh"][task])
    assert measured["max_absolute_prediction_difference"] < 1e-10
    assert measured["callable"]["call_evaluations"] > 0
    assert measured["callable"]["cache_hits"] > 0
    assert measured["callable"]["primitive_steps"] < measured["expanded"]["primitive_steps"]
    assert measured["expanded"]["call_evaluations"] == measured["expanded"]["cache_hits"] == 0
