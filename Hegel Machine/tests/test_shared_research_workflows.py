"""Existing library/checkpoint formats share one executable research-method book."""

import json

from hegel_machine.cognition.growth_methods import instantiate
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, TaskSpec
from hegel_machine.cognition.research_growth import GrowthResearcher
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def _json_roundtrip(value):
    return json.loads(json.dumps(value))


def _learned_library():
    library = KnowledgeLibrary()
    spec = TaskSpec("source", "Predict a response.", ("x",), (), (),
                    feature_bounds={"x": (0., 1.)})
    program = Program("feature", name="x")
    rows = [Observation({"x": x}, x) for x in (.1, .4, .7, .9)]
    library.learn(Candidate("source", program, "fit"), spec, rows)
    probe = Probe("acquire", tuple(row.features for row in rows))
    trace = [
        {"action": "vary_controls", "probe": probe},
        {"action": "fit", "recipe": {"estimator": "residual_expression_search"}},
        {"action": "confirm", "probe": probe},
    ]
    diagnostics = {"feature_count": 1, "residual_error": 0.,
                   "feature_bounds": spec.feature_bounds}
    method = library.research_workflows.learn(trace, diagnostics,
        success=True, task_id=spec.task_id)
    return library, method, diagnostics


def test_json_library_shares_replay_and_feedback_across_both_researchers():
    library, method, diagnostics = _learned_library()
    checkpoint = _json_roundtrip(GrowthResearcher(library=library).to_dict())
    unified = UnifiedResearchAgent(library=KnowledgeLibrary.from_dict(checkpoint["library"]))
    growth = GrowthResearcher(library=unified.library)
    assert growth.methods is unified.library.research_workflows

    proposed = unified.library.research_workflows.propose(diagnostics)
    target = TaskSpec("target", "Predict a response.", ("renamed",), (), (),
                      feature_bounds={"renamed": (-2., 2.)})
    actions = instantiate(proposed, target)
    assert [action["action"] for action in actions] == ["vary_controls", "fit", "confirm"]
    assert actions[0]["probe"].sequence[0] == {"renamed": -1.6}

    unified.library.research_workflows.record_outcome(method["method_id"], diagnostics,
        success=False, task_id="target")
    assert len(growth.methods.methods[0]["examples"]) == 2
    restored = GrowthResearcher(state=_json_roundtrip(growth.to_dict()))
    assert restored.methods.to_dict() == unified.library.research_workflows.to_dict()
    assert restored.to_dict()["methods"] == restored.to_dict()["library"]["research_workflows"]


def test_old_growth_json_checkpoint_moves_actual_top_level_methods_into_library():
    library, method, diagnostics = _learned_library()
    old_checkpoint = GrowthResearcher(library=library).to_dict()
    del old_checkpoint["library"]["research_workflows"]
    restored = GrowthResearcher(state=_json_roundtrip(old_checkpoint))
    assert restored.methods is restored.library.research_workflows
    assert restored.methods.propose(diagnostics)["method_id"] == method["method_id"]
    assert restored.library.research_workflows.to_dict() == old_checkpoint["methods"]


def test_embedded_or_explicit_library_is_authoritative_over_top_level_methods():
    library, _, _ = _learned_library()
    checkpoint = GrowthResearcher(library=library).to_dict()
    checkpoint["library"]["research_workflows"] = {"methods": []}
    assert not GrowthResearcher(state=_json_roundtrip(checkpoint)).methods.methods

    del checkpoint["library"]["research_workflows"]
    explicit = KnowledgeLibrary()
    restored = GrowthResearcher(library=explicit, state=_json_roundtrip(checkpoint))
    assert restored.methods is explicit.research_workflows
    assert not restored.methods.methods


def test_old_unified_library_json_loads_with_empty_workflow_book():
    library, _, _ = _learned_library()
    old_payload = library.to_dict()
    del old_payload["research_workflows"]
    unified = UnifiedResearchAgent(library=KnowledgeLibrary.from_dict(_json_roundtrip(old_payload)))
    growth = GrowthResearcher(library=unified.library)
    assert not unified.library.research_workflows.methods
    assert growth.methods is unified.library.research_workflows
    assert unified.library.to_dict()["concepts"] == old_payload["concepts"]
    assert unified.library.methods.to_dict() == old_payload["methods"]


def test_removing_concepts_preserves_independently_learned_workflows_without_aliasing():
    library, method, diagnostics = _learned_library()
    retained = library.without({library.concepts[0].concept_id})
    assert not retained.concepts
    assert retained.research_workflows.to_dict() == library.research_workflows.to_dict()
    retained.research_workflows.record_outcome(method["method_id"], diagnostics,
        success=False, task_id="target")
    assert len(retained.research_workflows.methods[0]["examples"]) == 2
    assert len(library.research_workflows.methods[0]["examples"]) == 1
