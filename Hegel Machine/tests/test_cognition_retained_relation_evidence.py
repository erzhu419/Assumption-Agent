"""Historical counterexamples keep their own response and can be revised locally."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from test_cognition_relation_trajectory_evidence import measured_law, mse


def physical_rows(distances, *, radius=.8, coordinate_scale=1., prefix="", consumer=False):
    """Independent exchange, reset event, and window readout, with public controls."""
    history, flows, events, responses, rows = [], [], [], [], []
    for distance in distances:
        fields = {prefix + key: value for key, value in {
            "f0": 1., "f1": 0., "x0": 0., "x1": distance, "q0": .2, "q1": 1.2}.items()}
        flow = .0015 if coordinate_scale * distance < radius else 0.
        events.append(0. if not flows else flow - flows[-1])
        flows.append(flow)
        responses.append(sum(events[-3:]))
        target = sum(responses[-2:]) + .2 * (.2 + 1.2) if consumer else responses[-1]
        rows.append(Observation(fields, target, tuple(history)))
        history.append({**fields, "__target__": target})
    return rows


def saved_source(*, coordinate_scale=1.):
    source = measured_law()
    distances = ([1.2, .7, .7, .7, .7, 1.2, 1.2, 1.2, 1.2] if coordinate_scale == 1.
                 else [.6, .3, .3, .3, .3, .6, .6, .6, .6])
    paid = physical_rows(distances, coordinate_scale=coordinate_scale)
    budget = SearchBudget(30000)
    research = RelationIntervalResearch(source, [], budget,
        structure={"fits": [{"radius": .8}], "radius_interval": [.2, 1.]})
    entries = research.alternatives(paid, budget, experiments=[paid])
    assert entries and all(mse(item["candidate"].program, paid) < 1e-16 for item in entries)
    selected = entries[len(entries) // 2]
    record = json.loads(json.dumps({"source_concept_id": "saved_source", "status": "supported",
                                   "research_program": selected["research_program"]}))
    return source, paid, selected, record


def consumer_call(selected, *, prefix="next_", coordinate_scale=1., outer=True):
    source = Program.from_dict(json.loads(json.dumps(selected["candidate"].program.to_dict())))
    names = tuple(sorted(source.required_features()))

    def parameters(node):
        if node.op == "feature":
            return Program("arg", name=node.name)
        return replace(node, children=tuple(parameters(child) for child in node.children))

    definition = Definition("saved_response", names, parameters(source), "saved_source")
    children = []
    for name in names:
        child = Program("feature", name=prefix + name)
        if name.startswith("x") and coordinate_scale != 1.:
            child = Program("mul", children=(Program("const", value=coordinate_scale), child))
        children.append(child)
    call = Program("call", name=definition.definition_id, definition=definition, children=tuple(children))
    if not outer:
        return call
    total = Program("object_sum", children=(Program("feature", name=prefix + "q0"),
                                             Program("feature", name=prefix + "q1")))
    return Program("add", children=(Program("window_sum", lag=2, children=(call,)),
        Program("mul", children=(Program("const", value=.2), total))))


def test_json_counterexamples_constrain_renamed_consumer_without_reusing_source_targets():
    source, old_paid, selected, record = saved_source()
    untouched = deepcopy((source.to_dict(), record))
    consumer = consumer_call(selected)
    paid = physical_rows([1.2, .1, .1, 1.2] * 3, prefix="next_", consumer=True)
    budget = SearchBudget(50000)
    inherited = RelationIntervalResearch(consumer, [record], budget)
    answers = inherited.alternatives(paid, budget)

    assert inherited.evidence_contexts and answers
    assert all(mse(item["candidate"].program, paid) < 1e-16 for item in answers)
    # The source's window/event target is near zero; this consumer adds a new
    # sum and window. Treating those historical labels as current labels fails.
    assert min(row.target for row in paid) > .27
    assert max(abs(row.target) for row in old_paid) < .01
    assert all(next(iter(item["parameter_values"].values())) > .7 for item in answers)
    held = physical_rows([1.2, .7, .7, .7, 1.2, 1.2, 1.2], prefix="next_", consumer=True)
    assert all(mse(item["candidate"].program, held) < 1e-16 for item in answers)
    assert (source.to_dict(), record) == untouched
    assert budget.spent <= budget.limit


def test_current_mechanism_counterexample_reopens_excluded_domain_without_erasing_source():
    source, old_paid, selected, record = saved_source()
    untouched = deepcopy((source.to_dict(), record))
    consumer = consumer_call(selected)
    changed = physical_rows([1.2, .6, .6, .6, .6, 1.2, .45, .45, .45, .45, 1.2, 1.2],
                            radius=.5, prefix="next_", consumer=True)
    assert mse(consumer, changed) > 1e-7
    budget = SearchBudget(70000)
    inherited = RelationIntervalResearch(consumer, [record], budget)
    answers = inherited.alternatives(changed, budget, experiments=[changed])

    assert answers and inherited.evidence_updates
    assert all(.45 < next(iter(item["parameter_values"].values())) < .6 for item in answers)
    assert all(mse(item["candidate"].program, changed) < 1e-16 for item in answers)
    assert (source.to_dict(), record) == untouched
    # Returning to the old mechanism restores the original saved interpretation,
    # including its independent trajectory; the changed task did not revoke it.
    returned = consumer_call(selected, prefix="return_", outer=False)
    returning_rows = physical_rows([1.2, .7, .7, .7, .7, 1.2, 1.2, 1.2], prefix="return_")
    original = RelationIntervalResearch(returned, [record], budget)
    again = original.alternatives(returning_rows, budget)
    assert again and not original.evidence_updates
    assert all(mse(item["candidate"].program, returning_rows) < 1e-16 for item in again)
    assert mse(selected["candidate"].program, old_paid) < 1e-16
    assert budget.spent <= budget.limit


def test_observed_input_revision_reinterprets_new_paid_rows_and_keeps_old_binding_record():
    source, old_paid, selected, record = saved_source(coordinate_scale=2.)
    untouched = deepcopy((source.to_dict(), record))
    # An actual coordinate reading now disproves the old identity calibration.
    # The same .3 control is measured as .6; no radius value is blacklisted.
    command = Observation({"next_x1": .3}, 0.)
    old_input = Program("feature", name="next_x1")
    corrected_input = Program("mul", children=(Program("const", value=2.), old_input))
    assert old_input.predict(command) != .6 and corrected_input.predict(command) == .6
    rebound = consumer_call(selected, coordinate_scale=2., outer=False)
    paid = physical_rows([.6, .3, .3, .3, .3, .6, .6, .6, .6],
                         coordinate_scale=2., prefix="next_")
    assert mse(rebound, paid) > 1e-7
    budget = SearchBudget(50000)
    inherited = RelationIntervalResearch(rebound, [record], budget)
    assert inherited.evidence_contexts
    assert inherited.reopen("observed_input_revision")
    answers = inherited.alternatives(paid, budget, experiments=[paid])

    assert answers
    assert all(next(iter(item["parameter_values"].values())) > .6 for item in answers)
    assert all(mse(item["candidate"].program, paid) < 1e-16 for item in answers)
    assert (source.to_dict(), record) == untouched
    assert mse(selected["candidate"].program, old_paid) < 1e-16
    assert budget.spent <= budget.limit
