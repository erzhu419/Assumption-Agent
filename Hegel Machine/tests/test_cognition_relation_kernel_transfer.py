"""Relation-kernel transfer stays below the complete response boundary."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.relation_kernel_priors import recall_relation_kernels
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from test_cognition_nested_relation_revisions import composition, mse
from test_cognition_retained_relation_intervals import observations


def _event_window(program, *, width=2):
    return Program("window_sum", lag=width, children=(
        Program("event_delta", lag=1, children=(program,)),))


def _focus_relation(radius=.6, *, order=(0, 1), role_swap=False):
    triples = []
    for index in order:
        roles = ("f", "x", "q")
        if role_swap:
            roles = ("f", "q", "x")
        triples.extend(Program("feature", name=f"{roles[role]}{index}")
                       for role in range(3))
    return Program("neighbor_sum", children=(
        Program("const", value=radius), *triples))


def replace_radius_with_arg(program):
    return replace(program, children=(Program("arg", name="radius"), *program.children[1:]))


def _structure(low=.5, high=.8, radius=.6):
    return {"fits": [{"radius": radius}], "radius_interval": [low, high]}


def _source_record(source_id="focus", interval=(.5, .7)):
    source = _focus_relation()
    paid = observations(source, (.1, .6, .9, 1.2) * 4)
    research = RelationIntervalResearch(source, [], SearchBudget(100000),
        structure=_structure(.5, .8))
    entries = research.alternatives(paid, SearchBudget(100000))
    assert entries
    record = json.loads(json.dumps({
        "source_concept_id": source_id,
        "status": "supported",
        "research_program": entries[-1]["research_program"],
    }))
    parameter = record["research_program"]["selection_steps"][-1]["parameters"][0]
    parameter["radius_intervals"] = [list(interval)]
    parameter["prior_radius_intervals"] = [list(interval)]
    return source, record


def test_focus_kernel_is_reused_inside_fresh_event_window_and_keeps_boundary_predictions():
    _, focus = _source_record()
    current = _event_window(composition(.6))
    truth = _event_window(composition(.68))
    rows = observations(truth, (.1, .6, .68, .9, 1.2) * 5)

    research = RelationIntervalResearch(current, [focus], SearchBudget(300000),
        structure=_structure())
    parameter = research.parameters[0]
    assert parameter["kernel_prior_status"] == "pending"
    assert [item["source_concept_id"] for item in parameter["kernel_priors"]] == ["focus"]

    entries = research.alternatives(rows, SearchBudget(300000), experiments=[rows])
    assert entries
    assert all(mse(entry["candidate"].program, rows) < 1e-7 for entry in entries)
    assert all(entry["method_source_ids"] == ["focus"] for entry in entries)
    assert all(entry["research_program"]["selection_steps"][-1]["parameters"][0]
               ["kernel_source_ids"] == ["focus"] for entry in entries)

    # The inherited kernel is a radius hypothesis, so alternatives still make
    # different predictions at an unpurchased boundary in the full consumer.
    boundary = observations(truth, (.62, .66, .72, 1.2))
    values = [[entry["candidate"].program.predict(row) for row in boundary]
              for entry in entries]
    assert max(max(column) - min(column) for column in zip(*values)) > .1


def test_a_narrow_pending_kernel_still_emits_a_consuming_candidate():
    _, focus = _source_record(interval=(.6000000, .6000005))
    current = _event_window(composition(.6))
    rows = observations(_event_window(composition(.6001)),
                        (.1, .6, .9, 1.2) * 4)
    research = RelationIntervalResearch(current, [focus], SearchBudget(300000),
        structure=_structure())

    assert research.parameters[0]["kernel_prior_status"] == "pending"
    entries = research.alternatives(rows, SearchBudget(300000))
    assert entries
    assert all(entry["method_source_ids"] == ["focus"] for entry in entries)
    assert all(entry["research_program"]["selection_steps"][-1]["parameters"][0]
               ["kernel_prior_status"] == "supported" for entry in entries)


def test_current_counterexample_withdraws_kernel_without_mutating_source_json():
    _, focus = _source_record(interval=(.5, .55))
    source_before = deepcopy(focus)
    current = _event_window(composition(.6))
    truth = _event_window(composition(.68))
    rows = observations(truth, (.1, .6, .68, .9, 1.2) * 5)
    research = RelationIntervalResearch(current, [focus], SearchBudget(300000),
        structure=_structure())

    entries = research.alternatives(rows, SearchBudget(300000), experiments=[rows])
    assert entries and research.evidence_updates
    assert research.evidence_updates[0]["reason"] == "current_relation_counterexample"
    assert research.parameters[0]["kernel_prior_status"] == "withdrawn"
    assert all("method_source_ids" not in entry for entry in entries)
    assert focus == source_before


@pytest.mark.parametrize("change", ["field", "predicate", "role_swap", "order"])
def test_changed_relation_kernel_does_not_recall_focus_prior(change):
    _, focus = _source_record()
    features = tuple(Program("feature", name=name) for name in
                     ("f0", "x0", "q0", "f1", "x1", "q1"))
    if change == "field":
        features = (*features[:-1], Program("feature", name="q_changed"))
        current = Program("neighbor_sum", children=(Program("arg", name="radius"), *features))
    elif change == "role_swap":
        features = tuple(Program("feature", name=name) for name in
                         ("f0", "q0", "x0", "f1", "q1", "x1"))
        current = Program("neighbor_sum", children=(Program("arg", name="radius"), *features))
    elif change == "order":
        current = _focus_relation(order=(1, 0))
        current = replace_radius_with_arg(current)
    else:
        predicate = Program("lt", children=(
            Program("arg", name="edge_target_quantity"),
            Program("arg", name="edge_source_quantity")))
        current = Program("relation_sum", name="right", children=(
            Program("arg", name="radius"), predicate, *features))
    parameters = [{"symbol": "radius", "source_concept_id": None}]
    recalled = recall_relation_kernels(current, parameters, [focus], SearchBudget(100000))
    if change == "order":
        assert recalled["radius"] and recalled["radius"][0]["source_concept_id"] == "focus"
    else:
        assert recalled == {"radius": []}


def test_conflicting_kernel_sources_remain_separate_until_paid_rows_choose_one():
    _, low = _source_record("focus_low", (.5000000, .5000005))
    _, high = _source_record("focus_high", (.7500000, .7500005))
    current = _event_window(composition(.6))
    quiet = observations(_event_window(composition(.7501)), (.1, .9) * 8)
    research = RelationIntervalResearch(current, [low, high], SearchBudget(400000),
        structure=_structure())

    sources = research.parameters[0]["kernel_priors"]
    assert {item["source_concept_id"] for item in sources} == {"focus_low", "focus_high"}
    assert research.parameters[0]["radius_intervals"] == [[.5000000, .5000005], [.7500000, .7500005]]
    quiet_entries = research.alternatives(quiet, SearchBudget(400000))
    assert quiet_entries
    assert {source for entry in quiet_entries for source in entry["method_source_ids"]} == {
        "focus_low", "focus_high"}
    assert research.questions(quiet_entries)

    paid = observations(_event_window(composition(.7501)), (.1, .75, .9) * 5)
    entries = research.alternatives([*quiet, *paid], SearchBudget(400000), experiments=[paid])
    assert entries
    assert all(entry["method_source_ids"] == ["focus_high"] for entry in entries)


def test_json_research_entry_restores_kernel_and_reopens_after_new_counterexample():
    _, focus = _source_record(interval=(.5, .55))
    current = _event_window(composition(.6))
    research = RelationIntervalResearch(current, [focus], SearchBudget(300000),
        structure=_structure())
    quiet_rows = observations(current, (.1, .6, .9, 1.2) * 4)
    entry = research.alternatives(quiet_rows, SearchBudget(300000))[0]
    saved = json.loads(json.dumps(entry["research_program"]))
    # The saved consuming program is the executable state whose own radius
    # agrees with the serialized parameter value; restoring against the old
    # pre-selection radius would correctly fail the exact procedure match.
    restored_program = Program.from_dict(saved["selection_steps"][-1]["program"])
    restored = RelationIntervalResearch(restored_program, [], SearchBudget(300000),
        research_program=saved)
    before = deepcopy(saved)

    changed = observations(_event_window(composition(.68)),
                           (.1, .6, .68, .9, 1.2) * 5)
    entries = restored.alternatives(changed, SearchBudget(300000), experiments=[changed])
    assert entries and restored.evidence_updates
    assert restored.parameters[0]["kernel_prior_status"] == "withdrawn"
    assert restored.evidence_updates[0]["reopened_intervals"][0]["radius_interval"] == [.5, .8]
    assert any(low <= .68 <= high for low, high in
               restored.parameters[0]["radius_intervals"])
    assert saved == before


def test_two_relation_parameters_are_adopted_without_prematurely_crediting_the_second():
    _, left_prior = _source_record("left", (.5000000, .5000005))
    _, right_prior = _source_record("right", (.7000000, .7000005))
    names = ("f0", "x0", "q0", "f1", "x1", "q1")
    rename = {name: "right_" + name for name in names}
    saved = right_prior["research_program"]["selection_steps"][-1]
    saved["parameter_template"] = Program.from_dict(saved["parameter_template"]).remap_features(rename).to_dict()
    relations = tuple(Program("neighbor_sum", children=(Program("arg", name=symbol),
        *(Program("feature", name=prefix + name) for name in names)))
        for symbol, prefix in (("left_radius", ""), ("right_radius", "right_")))
    template = Program("add", children=relations)
    current = Program("add", children=tuple(replace(relation,
        children=(Program("const", value=.6), *relation.children[1:])) for relation in relations))
    own = {"selection_steps": [{"operation": "resolve_relation_intervals",
        "parameter_template": template.to_dict(), "parameters": [
            {"symbol": symbol, "radius": .6, "radius_intervals": [[.4, .8]],
             "source_concept_id": None} for symbol in ("left_radius", "right_radius")]}]}
    quiet = []
    for index, distance in enumerate((.1, 1.2) * 6):
        point = dict(f0=1., f1=0., x0=0., x1=distance, q0=.1 * index, q1=1.)
        row = Observation({**point, **{rename[key]: value for key, value in point.items()}}, 0.)
        quiet.append(replace(row, target=current.predict(row)))
    research = RelationIntervalResearch(current, [left_prior, right_prior],
        SearchBudget(10000), research_program=own)
    first = research.alternatives(quiet, SearchBudget(10000))[0]
    research.adopt(first)
    assert first["method_source_ids"] == ["left"]
    assert [item["kernel_prior_status"] for item in research.parameters] == ["supported", "pending"]
    right_symbol = research.parameters[1]["symbol"]
    assert research.values[right_symbol] == .6
    second = research.alternatives(quiet, SearchBudget(10000))[0]
    research.adopt(second)
    assert second["method_source_ids"] == ["left", "right"]
    assert all(item["kernel_prior_status"] == "supported" for item in research.parameters)
    assert .7000000 <= research.values[right_symbol] <= .7000005
