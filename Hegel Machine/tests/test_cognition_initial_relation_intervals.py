"""First structural uncertainty stays compact and does not constrain a new law."""

from dataclasses import replace
import json

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.relation_revisions import _difference, _expose, _geometry, _relations
from test_cognition_nested_relation_revisions import composition, mse
from test_cognition_retained_relation_intervals import observations, prior


def test_first_structure_binds_its_radius_without_copying_an_unneeded_old_expression():
    parent = composition(.6).expanded()
    actual = composition(.7).expanded()
    original = parent.to_dict()
    structure = prior()["structure"]
    research = RelationIntervalResearch(parent, [], SearchBudget(3000), structure=structure)
    paid = observations(actual, (.1, .58, 1.2) * 5)
    candidates = research.alternatives(paid, SearchBudget(30000))
    assert candidates and all(item["candidate"].program.size == parent.size for item in candidates)
    selected = candidates[-1]
    assert mse(selected["candidate"].program, paid) < 1e-24
    assert parent.to_dict() == original

    # The compact first answer still exposes its unresolved parameter when the
    # stored program later becomes a callable concept with new input names.
    program = selected["candidate"].program
    names = tuple(sorted(program.required_features()))
    def parameterize(node):
        if node.op == "feature":
            return Program("arg", name=node.name)
        return replace(node, children=tuple(parameterize(child) for child in node.children))
    definition = Definition("initial:definition", names, parameterize(program), "initial")
    call = Program("call", name=definition.definition_id, definition=definition,
        children=tuple(Program("feature", name="later_" + name) for name in names))
    record = json.loads(json.dumps({"source_concept_id": "initial", "status": "supported",
                                    "research_program": selected["research_program"]}))
    inherited = RelationIntervalResearch(call, [record], SearchBudget(10000))
    assert inherited.intervals[0]["radius_interval"] == [.58, .7]
    later = inherited.alternatives(observations(composition(.7, "later_"),
        (.1, .58, 1.2) * 5, prefix="later_"), SearchBudget(30000))
    assert len(later) > 1
    assert all(item["candidate"].program == call or
               call in item["candidate"].program.children for item in later)


def test_new_paid_response_can_refute_the_entire_inherited_interval():
    parent = composition(.6).expanded()
    actual = composition(1.2).expanded()
    structure = prior()["structure"]
    research = RelationIntervalResearch(parent, [], SearchBudget(3000), structure=structure)
    paid = observations(actual, (.1, .9, 1.3, .9, .1) * 3)
    assert research.alternatives(paid, SearchBudget(30000)) == []
    assert research.status == "refuted"
    assert structure["radius_interval"] == [.5, .7]


def test_refining_an_existing_local_correction_keeps_its_exact_old_call_without_another_delta():
    old = composition(.6)
    opened = _expose(old, SearchBudget(3000))
    change = _difference(opened, _geometry(_relations(opened)[0]), 1.2)
    current = Program("add", children=(old, change)).simplified()
    original = old.to_dict()
    research = RelationIntervalResearch(current, [], SearchBudget(10000), research_program={
        "selection_steps": [{"operation": "revise_nested_relation", "radius": 1.2,
                              "radius_interval": [1.1, 1.3]}]})
    paid = observations(composition(1.25), (.1, 1.7) * 6)
    candidates = research.alternatives(paid, SearchBudget(30000))
    assert len(candidates) == 3
    assert all(item["candidate"].program.size == current.size for item in candidates)
    assert all(old in item["candidate"].program.children for item in candidates)
    assert old.to_dict() == original
    unobserved = observations(current, (.1, 1.2, 1.7, .1, .1))
    for item in candidates:
        radius = item["parameter_values"]["relation_parameter_0"]
        expected = composition(radius)
        assert all(abs(item["candidate"].program.predict(row) - expected.predict(row)) < 1e-12
                   for row in unobserved)
