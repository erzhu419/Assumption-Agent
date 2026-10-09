import importlib.util
from pathlib import Path

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec


spec = importlib.util.spec_from_file_location("growth_paper_study",
    Path(__file__).resolve().parents[1] / "scripts/run_growth_paper_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_deletion_interventions_change_only_their_persisted_knowledge():
    library = KnowledgeLibrary()
    source = Program("feature", name="x")
    rows = [Observation({"x": value}, value) for value in (-1., -.2, .4, .8)]
    task = TaskSpec("public_process", "Predict a response.", ("x",), (), (),
                    feature_bounds={"x": (-1., 1.)})
    library.learn(Candidate("learned", source, "fit"), task, rows)
    state = {"library": library.to_dict(), "methods": {"methods": [{"method_id": "m1"}], "next_method_index": 2},
             "abstractions": [{"definition_id": "a1"}], "tasks": [{"program": source.to_dict()}]}
    state["library"]["research_workflows"] = state["methods"]
    programs_removed = study.delete_programs(state)
    methods_removed = study.delete_methods(state)
    assert not programs_removed["library"]["concepts"]
    assert not programs_removed["abstractions"]
    assert programs_removed["methods"] == state["methods"]
    assert methods_removed["methods"] == {"methods": [], "next_method_index": 2}
    assert methods_removed["library"]["research_workflows"] == {"methods": [], "next_method_index": 2}
    assert {key: value for key, value in methods_removed["library"].items()
            if key != "research_workflows"} == {
                key: value for key, value in state["library"].items()
                if key != "research_workflows"}
    assert methods_removed["abstractions"] == state["abstractions"]
    assert state["library"]["concepts"] and state["methods"]["methods"]


def test_static_scope_contrast_uses_the_identified_old_branch():
    old = Program("feature", name="x")
    new = Program("mul", children=(Program("const", value=2.), old))
    learned = Program("if", children=(Program("lt", children=(old, Program("const", value=0.))), old, new))
    rows = [Observation({"x": value}, value if value < 0 else 2 * value)
            for value in (-.9, -.3, .4, .7)]
    counterfactual, cost = study._scope_counterfactual(old, learned, rows)
    assert counterfactual == new
    assert cost > 0
    unavailable, _ = study._scope_counterfactual(old, new, rows)
    assert unavailable is None
