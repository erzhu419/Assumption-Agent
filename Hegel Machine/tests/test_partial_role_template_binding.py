"""Sparse induced roles must remain usable without inventing missing roles."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import TaskSpec
from hegel_machine.cognition.partial_object_knowledge import partial_object_schema
from hegel_machine.cognition.scale_bindings import aggregate_measurement_binding, aggregate_role_binding


def template(source, role):
    return AbstractTemplate(f"{source}:abstract:0", source,
        Program("feature", name="argument"), ("argument",), (f"old_{role}",), (),
        object_roles={"argument": (0, role)})


def partial_model():
    return {"complete": False, "schema": None, "focus_control": None,
        "groups": [{"object_id": name, "q": f"{name}_q", "u": f"{name}_u",
                    "controls": {"q": f"control_{name}_q", "u": f"control_{name}_u"}}
                   for name in ("a", "b")],
        "unresolved": ["focal membership", "position ownership"]}


def test_typed_binding_skips_only_assignments_missing_the_required_role():
    schema = partial_object_schema(partial_model())
    fields = ("a_q", "a_u", "b_q", "b_u", "unassigned_focus_sensor")
    focal = template("focal", "f")
    assert list(focal.mappings(fields, object_schema=schema)) == []
    # The 010 source keeps the original instrument names. Its valid identity
    # must survive requesting a second, currently unbound, object assignment.
    assert list(focal.mappings((*fields, "old_f"), object_schema=schema)) == [
        {"argument": "old_f"}]
    schema["fields"]["b"]["f"] = "unassigned_focus_sensor"
    assert list(focal.mappings(fields, object_schema=schema)) == [
        {"argument": "unassigned_focus_sensor"}]
    assert list(template("quantity", "q").mappings(fields, object_schema=schema)) == [
        {"argument": "a_q"}, {"argument": "b_q"}]


@pytest.mark.parametrize("organization", ["archive", "fixed_pr2", "pr3"])
def test_ordinary_replay_retains_quantity_method_and_defers_unbound_focal_method(tmp_path, organization):
    model = partial_model()
    before = deepcopy(model)
    fields = tuple(group[role] for group in model["groups"] for role in ("q", "u"))
    spec = TaskSpec("partial", "Current paid role induction", fields, (), (),
                    context={"object_model": model})
    agent = CommonObjectResearchAgent(workspace=tmp_path, organization=organization)
    source_schema = {**partial_object_schema(model), "fields": {
        name: {role: f"old_{name}_{role}" for role in ("q", "u", "x", "f")}
        for name in ("a", "b")}}
    calls = []
    for source, role in (("quantity", "q"), ("focal", "f")):
        agent.library.abstracts.append(template(source, role))
        agent.library.meta_priors.append({"source_concept_id": source,
            "object_schema": source_schema, "research_program": {"kind": "retained_method"}})
        definition = Definition(f"definition:{source}", (), Program("const", value=0.), source)
        calls.append(Program("call", name=definition.definition_id, definition=definition))
    old = deepcopy(agent.library.meta_priors)
    plans = agent._replay_program_queue(spec, calls)
    assert [plan[0] for plan in plans] == ["quantity"]
    assert plans[0][3] == [{"argument": "a_q"}, {"argument": "b_q"}]
    assert agent.library.meta_priors == old and model == before
    assert not model["complete"]


def test_aggregate_binding_requires_only_the_requested_established_roles():
    target = partial_object_schema(partial_model())
    source = {"objects": ["whole"], "fields": {"whole": {"q": "old_q", "f": "old_f"}},
        "members": {"whole": ["a", "b"]}, "controls": target["controls"], "focus_control": None}
    before = deepcopy((source, target))
    assert aggregate_measurement_binding(source, target) is None
    assert aggregate_role_binding(template("focal", "f"), source, target) is None
    mapping = aggregate_role_binding(template("quantity", "q"), source, target)
    assert mapping["argument"].required_features() == {"a_q", "b_q"}
    assert (source, target) == before
