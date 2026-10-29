"""A completed same-scene radius partition avoids repeating paid geometry."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from hegel_machine.cognition import relation_uncertainty as module
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch


def law(radius):
    neighbor = Program('neighbor_sum', children=(Program('const', value=radius),
        *(Program('feature', name=name) for name in ('f0', 'x0', 'q0', 'f1', 'x1', 'q1'))))
    return Program('window_sum', lag=3, children=(Program('event_delta', lag=1,
                                                      children=(neighbor,)),))


def rows(program, distances):
    history, observations = [], []
    for index, distance in enumerate(distances):
        fields = dict(f0=1., x0=0., q0=.1 * index, f1=0., x1=distance, q1=1. + .07 * index)
        row = Observation(fields, 0., tuple(history))
        actual = replace(row, target=program.predict(row))
        observations.append(actual)
        history.append({**fields, '__target__': actual.target})
    return observations


def saved_partition():
    current = law(.8)
    paid = rows(current, (.1, 1.2) * 8)
    research = RelationIntervalResearch(current, [], SearchBudget(10000),
        structure={'fits': [{'radius': .8}], 'radius_interval': [.4, 1.]})
    answers = research.alternatives(paid, SearchBudget(10000), experiments=[paid])
    selected = next(item for item in answers if item['candidate'].program == current)
    return paid, decode_entry(json.loads(json.dumps(encode_entry(selected))))


def restore(entry):
    return RelationIntervalResearch(entry['candidate'].program, [], SearchBudget(10000),
                                    research_program=entry['research_program'])


def scientific_values(entries):
    return [(item['candidate'].program.to_dict(), item['parameter_values'],
             item['validation_loss'], item['research_program']['selection_steps'][-1]['parameters'])
            for item in entries]


def test_paid_partition_reuses_geometry_but_preserves_all_representatives_and_history(monkeypatch):
    paid, selected = saved_partition()
    acquired = rows(law(.8), (1.2, .7, .7, .7, 1.2))
    current = [*paid, *acquired]
    ordinary, resumed = restore(selected), restore(selected)
    assert resumed.reuse_paid_partition(paid, selected)
    original = module._distances
    calls = []

    def measured(relations, frames, budget):
        frames = list(frames)
        calls.append(frames)
        return original(relations, frames, budget)

    monkeypatch.setattr(module, '_distances', measured)
    ordinary_budget, resumed_budget = SearchBudget(100000), SearchBudget(100000)
    baseline = ordinary.alternatives(current, ordinary_budget, experiments=[paid, acquired])
    expected_frames = len(calls[-1])
    result = resumed.alternatives(current, resumed_budget, experiments=[paid, acquired])
    new_frames = len(calls[-1])
    assert scientific_values(result) == scientific_values(baseline)
    assert expected_frames - new_frames == len(paid)
    assert resumed_budget.spent == ordinary_budget.spent - 6 * len(paid)
    assert resumed.reuse_statistics['geometry_paid_rows_reused'] == len(paid)
    assert resumed.reuse_statistics['inherited_predictions_reused'] > 0
    assert any(frame.history for frame in calls[-1])
    assert all(item['radius_interval'] == [.7, 1.] for item in resumed.intervals)


def test_ordinary_consecutive_passes_retain_the_partition_after_adopting_a_new_radius():
    original = law(.8)
    paid = rows(original, (.1, 1.2) * 8)
    acquired = rows(original, (1.2, .7, .7, .7, 1.2))
    structure = {'fits': [{'radius': .8}], 'radius_interval': [.4, 1.]}
    ordinary = RelationIntervalResearch(original, [], SearchBudget(10000), structure=structure)
    baseline = RelationIntervalResearch(original, [], SearchBudget(10000), structure=deepcopy(structure))
    for research in (ordinary, baseline):
        answers = research.alternatives(paid, SearchBudget(10000), experiments=[paid])
        selected = next(item for item in reversed(answers) if item['candidate'].program != original)
        research.adopt(selected)
        assert research.program == original
        assert research.reuse_statistics['partition_rows_bound'] == len(paid)
    # The control deliberately repeats old geometry; the ordinary path uses
    # only alternatives -> adopt -> appended alternatives, without a restore API.
    baseline._paid_partition = None
    repeated_cost, incremental_cost = SearchBudget(10000), SearchBudget(10000)
    expected = baseline.alternatives([*paid, *acquired], repeated_cost, experiments=[paid, acquired])
    actual = ordinary.alternatives([*paid, *acquired], incremental_cost, experiments=[paid, acquired])
    assert scientific_values(actual) == scientific_values(expected)
    assert repeated_cost.spent - incremental_cost.spent == 6 * len(paid)
    assert ordinary.reuse_statistics['geometry_paid_rows_reused'] == len(paid)
    assert ordinary.reuse_statistics['partition_rows_bound'] == len(paid) + len(acquired)
    assert ordinary.program == original


@pytest.mark.parametrize('change', ('program', 'template', 'source', 'count'))
def test_restore_rejects_a_different_completed_contract(change):
    paid, selected = saved_partition()
    restored = restore(selected)
    supplied = deepcopy(selected)
    if change == 'program':
        supplied['candidate'] = replace(supplied['candidate'], program=law(.75))
    elif change == 'template':
        supplied['research_program']['selection_steps'][-1]['parameter_template'] = law(.9).to_dict()
    elif change == 'source':
        supplied['research_program']['selection_steps'][-1]['parameters'][0]['source_concept_id'] = 'other_source'
    else:
        supplied['research_program']['fit_rows'] += 1
    assert not restored.reuse_paid_partition(paid, supplied)
    assert restored.reuse_statistics['partition_rows_bound'] == 0


@pytest.mark.parametrize('change', ('prefix', 'in_place_history', 'template', 'source', 'reopened_domain', 'other_parameter'))
def test_geometry_reuse_stops_when_its_bound_scene_changes(change):
    paid, selected = saved_partition()
    restored = restore(selected)
    if change == 'other_parameter':
        second = {**deepcopy(restored.parameters[0]), 'symbol': 'other_parameter'}
        restored.parameters.append(second)
        restored.values['other_parameter'] = .6
        selected = deepcopy(selected)
        selected['parameter_values']['other_parameter'] = .6
        selected['research_program']['selection_steps'][-1]['parameters'].append(second)
    assert restored.reuse_paid_partition(paid, selected)
    current = [*paid, *rows(law(.8), (.7,))]
    if change == 'prefix':
        current[0] = replace(current[0], target=current[0].target + 1.)
    elif change == 'in_place_history':
        paid[-1].history[0]['x1'] = .62
    elif change == 'template':
        restored.template = Program('add', children=(restored.template, Program('const', value=.1)))
    elif change == 'source':
        restored.parameters[0]['source_concept_id'] = 'different_source'
    elif change == 'reopened_domain':
        restored.parameters[0]['radius_intervals'] = [[0., 2.]]
    else:
        restored.values['other_parameter'] = .65
    assert restored._new_geometry_rows(current, restored.parameters[0]) is current
    assert restored.reuse_statistics['geometry_paid_rows_reused'] == 0


def test_only_the_parameter_actually_partitioned_by_the_completed_pass_can_skip_geometry():
    paid, selected = saved_partition()
    restored = restore(selected)
    second = {**deepcopy(restored.parameters[0]), 'symbol': 'other_parameter'}
    restored.parameters.append(second)
    restored.values['other_parameter'] = .6
    selected['parameter_values']['other_parameter'] = .6
    selected['research_program']['selection_steps'][-1]['parameters'].append(second)
    assert restored.reuse_paid_partition(paid, selected)
    assert restored._new_geometry_rows(paid, restored.parameters[0]) == []
    assert restored._new_geometry_rows(paid, restored.parameters[1]) is paid


def test_inherited_response_and_exact_observation_are_required_for_score_reuse(monkeypatch):
    paid, selected = saved_partition()
    restored = restore(selected)
    values = dict(restored.values)
    errors = {id(row): 0. for row in paid}
    original = Program.predict
    calls = []

    def predict(program, row, *args, **kwargs):
        calls.append(row)
        return original(program, row, *args, **kwargs)

    monkeypatch.setattr(Program, 'predict', predict)
    budget = SearchBudget(1000)
    assert restored._supports_inherited_evidence(values, budget, 1e-7,
        evaluated_program=selected['candidate'].program, evaluated_rows=paid, evaluated_errors=errors)
    assert budget.spent == 0 and calls == []
    assert restored.reuse_statistics['inherited_predictions_reused'] == len(paid)
    assert restored._supports_inherited_evidence(values, budget, 1e-7,
        evaluated_program=Program('const', value=0.), evaluated_rows=paid, evaluated_errors=errors)
    assert budget.spent == len(paid) and len(calls) == len(paid)
    calls.clear()
    changed = restored._inherited_contexts[0]['trajectories'][0][0]
    changed['target'] += 1.
    assert not restored._supports_inherited_evidence(values, budget, 1e-7,
        evaluated_program=selected['candidate'].program, evaluated_rows=paid, evaluated_errors=errors)
    assert len(calls) == 1 and budget.spent == len(paid) + 1


def test_sparse_new_response_keeps_preparation_geometry_from_its_complete_history():
    paid, selected = saved_partition()
    actual = rows(law(.8), (1.2, .7, .7, .7, 1.2))
    new = actual[-1:]
    restored = restore(selected)
    assert restored.reuse_paid_partition(paid, selected)
    candidates = restored.alternatives([*paid, *new], SearchBudget(10000), experiments=[new])
    assert candidates and all(item['radius_interval'] == [.7, 1.] for item in restored.intervals)
    assert all(abs(item['candidate'].program.predict(new[0]) - new[0].target) < 1e-7 for item in candidates)
