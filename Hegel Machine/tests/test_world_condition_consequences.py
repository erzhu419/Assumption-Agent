"""Finite target consequences preserve grid boundaries and frozen transforms."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.world_response_conditions import (
    _experiments,
    _target_consequences,
    _target_projection,
    synthesize_response_conditions,
)
from test_world_response_conditions import band_packet


def constant(value):
    return Program('const', value=value)


def condition(name, lower, upper, *, inclusive=True):
    coordinate = Program('feature', name=name)
    right = (Program('ge', children=(constant(upper), coordinate)) if inclusive else
             Program('lt', children=(coordinate, constant(upper))))
    return Program('and', children=(Program('ge', children=(coordinate, constant(lower))), right))


def prepared(*, origin=0., residual=False):
    name = 'residual_0' if residual else 'frequency'
    derivations = {'residual_0': Program('mul', children=(constant(10.),
        Program('feature', name='density'))).to_dict()} if residual else {}
    programs = [condition(name, 1., 2.), condition(name, 2., 3., inclusive=False)]
    candidates = [{'id': str(index), 'status': 'compatible', 'program': program.to_dict(),
                   'feature_derivations': deepcopy(derivations)} for index, program in enumerate(programs)]
    result = {'candidates': candidates, 'selected_experiment': {
        'number': 2, 'prediction': 11.5 if residual else 1.5,
        'predictions': {'0': True, '1': False}}}
    inputs = {'targets': [5], 'measurements': {'5': {'density': 2.}},
              'bound_rows': {'5': {'density': 2.}},
              'control_grids': {'5': {'origin': origin, 'step': .5, 'lower': 0., 'upper': 10000.}}}
    return result, inputs


@pytest.mark.parametrize('origin,expected,counts', [(0., [[2, 5]], (2, 3)), (.25, [[2, 5]], (2, 2))])
def test_target_projection_counts_strict_and_inclusive_grid_boundaries_without_grid_sweep(origin, expected, counts):
    result, inputs = prepared(origin=origin)
    budget = SearchBudget(500)
    _target_consequences(result, inputs, budget)
    preview = result['target_consequences']
    target, = preview['targets']
    assert preview['status'] == target['status'] == 'complete'
    assert target['number'] == 5 and target['support'] == {'intervals': expected, 'grid_count': 4}
    assert tuple(item['support']['grid_count'] for item in target['by_hypothetical_outcome']) == counts
    assert target['guaranteed_search_reduction'] == 4 - max(counts)
    assert target['partition_cells'] < 20 and budget.spent < 500
    assert 'unconfirmed_finite_rules' in preview['scope']


def test_target_projection_uses_original_residual_transform_on_new_object():
    result, inputs = prepared(residual=True)
    _target_consequences(result, inputs, SearchBudget(500))
    target, = result['target_consequences']['targets']
    assert target['support'] == {'intervals': [[42, 45]], 'grid_count': 4}
    after = target['by_hypothetical_outcome']
    assert after[0]['support']['intervals'] == [[44, 45]]
    assert after[1]['support']['intervals'] == [[42, 44]]


def test_target_projection_budget_failure_is_unresolved_not_empty_support():
    result, inputs = prepared()
    _target_consequences(result, inputs, SearchBudget(1))
    target, = result['target_consequences']['targets']
    assert target['status'] == 'unresolved' and target['reason'] == 'projection_compute_limit'
    assert 'support' not in target and 'guaranteed_search_reduction' not in target


def test_target_projection_does_not_claim_complete_support_for_nonlinear_control_predicate():
    result, inputs = prepared()
    frequency = Program('feature', name='frequency')
    result['candidates'][0]['program'] = Program('ge', children=(
        Program('mul', children=(frequency, frequency)), constant(4.))).to_dict()
    _target_consequences(result, inputs, SearchBudget(500))
    target, = result['target_consequences']['targets']
    assert target['status'] == 'unresolved' and target['reason'] == 'nonaffine_control_boundary'
    assert 'support' not in target


@pytest.mark.parametrize('density,expected', [
    (2., {'intervals': [[42, 44]], 'grid_count': 3}),
    (2000., {'intervals': [], 'grid_count': 0}),
])
def test_unanimous_rules_still_project_reachable_target_support_without_a_discriminator(density, expected):
    result, inputs = prepared(residual=True)
    result['selected_experiment'] = None
    # Two retained rules agree everywhere on this target. Agreement can mean
    # reachable success or failure throughout its observed control grid.
    result['candidates'][1]['program'] = deepcopy(result['candidates'][0]['program'])
    inputs['measurements']['5']['density'] = density
    inputs['bound_rows']['5']['density'] = density
    candidates_before = deepcopy(result['candidates'])

    _target_consequences(result, inputs, SearchBudget(500))

    preview = result['target_consequences']
    target, = preview['targets']
    assert preview['status'] == target['status'] == 'complete'
    assert target['support'] == expected
    assert preview.get('experiment') is None
    assert not target.get('by_hypothetical_outcome')
    assert 'guaranteed_search_reduction' not in target
    assert result['selected_experiment'] is None
    assert result['candidates'] == candidates_before


def test_projection_without_a_discriminator_keeps_budget_exhaustion_distinct_from_empty_support():
    result, inputs = prepared()
    result['selected_experiment'] = None

    _target_consequences(result, inputs, SearchBudget(1))

    preview = result['target_consequences']
    target, = preview['targets']
    assert preview['status'] == target['status'] == 'unresolved'
    assert target['reason'] == 'projection_compute_limit'
    assert 'support' not in target
    assert preview.get('experiment') is None


def test_saved_compatible_pool_gains_missing_projection_once_without_regenerating_rules(monkeypatch):
    packet = band_packet()
    packet['targets'] = [1]
    saved = synthesize_response_conditions(packet)
    # A finite retained pool can contain just one compatible rule: it has no
    # disagreement, but still predicts reachable activation at a paid positive.
    saved['candidates'] = [next(item for item in saved['candidates'] if item['status'] == 'compatible')]
    saved.update(selected_experiment=None, experiment_action=None,
                 status='compatible_unconfirmed_conditions', regenerated=False)
    saved.pop('target_consequences')
    candidates_before = deepcopy(saved['candidates'])

    def unexpected_generation(*args, **kwargs):
        pytest.fail('unchanged paid evidence must reuse the saved compatible rules')

    monkeypatch.setattr('hegel_machine.cognition.world_response_conditions.synthesize', unexpected_generation)
    refreshed = synthesize_response_conditions(packet, previous=saved)
    repeated = synthesize_response_conditions(packet, previous=refreshed)

    target, = refreshed['target_consequences']['targets']
    assert target['status'] == 'complete'
    assert any(left <= 20 <= right for left, right in target['support']['intervals'])
    assert refreshed['compute_cost'] > 0 and repeated['compute_cost'] == 0
    assert refreshed['cached'] and repeated['cached']
    assert not refreshed['regenerated'] and not repeated['regenerated']
    assert refreshed['candidates'] == repeated['candidates'] == candidates_before
    assert repeated['target_consequences'] == refreshed['target_consequences']
    assert 'target_consequences' not in saved


def test_same_input_resume_completes_only_budget_limited_target_projection_then_caches(monkeypatch):
    packet = band_packet()
    packet['targets'] = [1, 2]
    first = synthesize_response_conditions(packet)
    # Retain two affine residual bands; nonlinear projection limits are a
    # different unresolved case from the exhausted allowance exercised here.
    compatible = [item for item in first['candidates'] if item['status'] == 'compatible'][:2]
    first['candidates'] = compatible
    first['experiment_search_complete'] = True
    # Reproduce a shared research allowance that reaches projection with enough
    # work left for the first target, but only one unit for the second target.
    first_target_budget = SearchBudget(first['compute_limit'])
    first_target = _target_projection(1, compatible, first['inputs'],
                                     first['selected_experiment'], first_target_budget)
    assert first_target['status'] == 'complete'
    shared_budget = SearchBudget(first['compute_limit'])
    shared_budget.spend(shared_budget.remaining - first_target_budget.spent - 1)
    _target_consequences(first, first['inputs'], shared_budget)
    complete, pending = first['target_consequences']['targets']
    assert complete == first_target
    assert pending['status'] == 'unresolved' and pending['reason'] == 'projection_compute_limit'
    assert 'support' not in pending
    candidates_before = deepcopy(first['candidates'])
    retried_targets = []

    def unexpected_search(*args, **kwargs):
        pytest.fail('same-input projection continuation must retain already-screened hypotheses')

    def tracked_projection(number, *args, **kwargs):
        retried_targets.append(number)
        return _target_projection(number, *args, **kwargs)

    monkeypatch.setattr('hegel_machine.cognition.world_response_conditions.synthesize', unexpected_search)
    monkeypatch.setattr('hegel_machine.cognition.world_response_conditions._screen', unexpected_search)
    monkeypatch.setattr('hegel_machine.cognition.world_response_conditions._target_projection', tracked_projection)
    resumed = synthesize_response_conditions(packet, previous=first)

    assert resumed['target_consequences']['status'] == 'complete'
    assert retried_targets == [2]
    assert resumed['target_consequences']['targets'][0] == complete
    assert resumed['target_consequences']['targets'][1]['support']['grid_count'] > 0
    assert 0 < resumed['compute_cost'] <= first['compute_limit']
    assert resumed['candidates'] == candidates_before
    retried_targets.clear()
    cached = synthesize_response_conditions(packet, previous=resumed)
    assert cached['cached'] and cached['compute_cost'] == 0
    assert not retried_targets
    assert cached['target_consequences'] == resumed['target_consequences']
    assert cached['candidates'] == candidates_before


def test_same_input_resume_finishes_experiment_search_and_replaces_old_unconditional_projection(monkeypatch):
    packet = band_packet()
    packet['targets'] = [1, 2]
    first = synthesize_response_conditions(packet)
    first['candidates'] = [item for item in first['candidates'] if item['status'] == 'compatible'][:2]
    first.update(selected_experiment=None, experiment_action=None,
                 experiment_count=0, status='compatible_unconfirmed_conditions')
    _experiments(first, first['inputs'], SearchBudget(1))
    assert first['experiment_search_complete'] is False
    assert first['selected_experiment'] is None
    # A separately available target-support calculation has no conditional
    # outcomes yet. Finishing the experiment stage must update that view too.
    _target_consequences(first, first['inputs'], SearchBudget(first['compute_limit']))
    assert first['target_consequences']['status'] == 'complete'
    assert first['target_consequences']['experiment'] is None
    candidates_before = deepcopy(first['candidates'])
    experiment_calls, projected_targets = [], []

    def unexpected_search(*args, **kwargs):
        pytest.fail('continuing experiment choice must not regenerate or re-screen frozen rules')

    def tracked_experiments(*args, **kwargs):
        experiment_calls.append(True)
        return _experiments(*args, **kwargs)

    def tracked_projection(number, *args, **kwargs):
        projected_targets.append(number)
        return _target_projection(number, *args, **kwargs)

    monkeypatch.setattr('hegel_machine.cognition.world_response_conditions.synthesize', unexpected_search)
    monkeypatch.setattr('hegel_machine.cognition.world_response_conditions._screen', unexpected_search)
    monkeypatch.setattr('hegel_machine.cognition.world_response_conditions._experiments', tracked_experiments)
    monkeypatch.setattr('hegel_machine.cognition.world_response_conditions._target_projection', tracked_projection)
    resumed = synthesize_response_conditions(packet, previous=first)

    assert resumed['experiment_search_complete'] is True
    assert experiment_calls == [True] and projected_targets == [1, 2]
    trial = resumed['selected_experiment']
    assert trial is not None and set(trial['predictions'].values()) == {False, True}
    preview = resumed['target_consequences']
    assert preview['status'] == 'complete'
    assert preview['experiment'] == {key: trial[key] for key in ('number', 'prediction')}
    assert all(target['by_hypothetical_outcome'] for target in preview['targets'])
    assert resumed['candidates'] == candidates_before and resumed['compute_cost'] > 0
    experiment_calls.clear()
    projected_targets.clear()
    cached = synthesize_response_conditions(packet, previous=resumed)
    assert cached['compute_cost'] == 0 and not experiment_calls and not projected_targets
    assert cached['candidates'] == candidates_before
    assert cached['selected_experiment'] == trial and cached['target_consequences'] == preview


def test_same_input_cache_does_not_retry_a_nonaffine_projection_with_more_of_the_same_budget(monkeypatch):
    packet = band_packet()
    packet['targets'] = [1]
    first = synthesize_response_conditions(packet)
    candidate = next(item for item in first['candidates'] if item['status'] == 'compatible')
    residual = Program('feature', name='residual_0')
    candidate['program'] = Program('ge', children=(constant(1.),
        Program('mul', children=(residual, residual)))).to_dict()
    candidate['claim'] = 'Activation iff the squared residual is at most one'
    first.update(candidates=[candidate], selected_experiment=None, experiment_action=None,
                 experiment_search_complete=True, status='compatible_unconfirmed_conditions')
    _target_consequences(first, first['inputs'], SearchBudget(first['compute_limit']))
    target, = first['target_consequences']['targets']
    assert target['status'] == 'unresolved' and target['reason'] == 'nonaffine_control_boundary'

    def unexpected_retry(*args, **kwargs):
        pytest.fail('an unsupported boundary is not repaired by repeating the same completed search')

    monkeypatch.setattr('hegel_machine.cognition.world_response_conditions._target_projection', unexpected_retry)
    monkeypatch.setattr('hegel_machine.cognition.world_response_conditions._experiments', unexpected_retry)
    cached = synthesize_response_conditions(packet, previous=first)
    assert cached['compute_cost'] == 0 and cached['cached']
    assert cached['candidates'] == first['candidates']
    assert cached['target_consequences'] == first['target_consequences']
