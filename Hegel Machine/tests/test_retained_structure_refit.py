"""Frozen witnesses refit their saved coordinates rather than repeat a search."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import native_condition_research as native
from hegel_machine.cognition import structural_inquiry
from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint, write_json_checkpoint
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.research_programs import _plan, replay_research_program
from test_structural_scientific_return import paid_scene


def saved_response():
    current = Program('feature', name='u')
    window = Program('window_sum', lag=2, children=(current,))
    old = Program('add', children=(current, window))
    step = {'operation': 'discover_conditional_coordinates',
        'question': 'Synthetic retained branch coordinates', 'program': old, 'loss': 1.,
        'gate_coordinate': current.to_dict(), 'threshold_interval': [.4, .5],
        'branch_basis': [current.to_dict()], 'width': 2, 'placement': 'inside_window'}
    payload = _plan({'program': old, 'loss': 1., 'terms': (current, window),
        'fit_terms': (current, window), 'coefficients': [0., 1., 1.]}, [step], 16, 80)
    history, rows = [], []
    for value in [-.7, .2, .8, -.3, .4, -.9, .1, .6] * 2:
        point = {'u': value}
        target = 2. + 3. * value + .4 * sum(p['u'] for p in [*history, point][-2:])
        rows.append(Observation(point, target, tuple(deepcopy(history))))
        history.append(point)
    return old, payload, rows


def test_fixed_scope_really_refits_parameters_without_conditional_search(monkeypatch):
    from hegel_machine.cognition import conditional_coordinates
    old, payload, rows = saved_response()
    before = deepcopy(payload)
    monkeypatch.setattr(conditional_coordinates, 'replay_conditional_coordinates',
        lambda *a, **k: pytest.fail('fixed-structure replay restarted gate/window search'))
    budget = SearchBudget(64)
    claim = replay_research_program(payload, rows, budget,
                                   execution_scope='retained_coordinates')
    assert claim['candidate'].program != old
    assert claim['training_loss'] < 1e-10
    assert budget.spent == len(rows) * (2 + 2) == 64
    assert budget.completed_fits == 1 and not budget.denied_requests
    assert payload == before
    assert claim['research_program']['execution_scope'] == 'retained_coordinates'
    assert claim['research_program']['selection_steps'][-1]['retained_structure']['width'] == 2


def test_quote_and_unfunded_group_do_not_execute_a_partial_prefix(monkeypatch):
    _, payload, rows = saved_response()
    sources = [{'research_program': payload,
                'source_ref': {'file': 'paid.json', 'json_path': [index]}} for index in range(2)]
    quote = native.retained_structure_refit_quote(sources, len(rows))
    assert quote['compute_upper'] == 128
    budget = SearchBudget(127)
    monkeypatch.setattr(native, 'replay_research_program',
        lambda *a, **k: pytest.fail('unfunded complete witness set started fitting'))
    cache = []
    outputs, summary = native._refit_saved_structures(sources, rows, budget,
        fit_evidence=list(range(16)), observed_record_count=1, risk_quantile=.5, cache=cache)
    assert outputs == cache == [] and budget.spent == 0
    assert not summary['complete']
    assert summary['blocked_work']['additional_compute'] == 1


def test_paid_saved_refits_reuse_exact_input_and_new_paid_rows_invalidate(monkeypatch):
    _, payload, rows = saved_response()
    sources = [{'research_program': payload,
                'source_ref': {'file': 'paid.json', 'json_path': [0]}}]
    original = deepcopy(sources)
    cache = []
    first = SearchBudget(64)
    outputs, summary = native._refit_saved_structures(sources, rows, first,
        fit_evidence=list(range(16)), observed_record_count=1, risk_quantile=.5, cache=cache)
    assert summary['complete'] and first.completed_fits == 1
    replay = native.replay_research_program
    monkeypatch.setattr(native, 'replay_research_program',
        lambda *a, **k: pytest.fail('recovery repeated a fully paid witness fit'))
    budget = SearchBudget(0)
    reused, summary = native._refit_saved_structures(sources, rows, budget,
        fit_evidence=list(range(16)), observed_record_count=1, risk_quantile=.5, cache=cache)
    assert reused == outputs and budget.spent == 0
    assert summary['complete'] and summary['attempts'][0]['cache_hit']
    assert summary['attempts'][0]['prior_paid_compute'] == 64
    monkeypatch.setattr(native, 'replay_research_program', replay)
    newer = [*rows, Observation({'u': .3}, 3.2, ({'u': -.9},))]
    budget = SearchBudget(68)
    _, summary = native._refit_saved_structures(sources, newer, budget,
        fit_evidence=list(range(17)), observed_record_count=2, risk_quantile=.5, cache=cache)
    assert summary['complete'] and not summary['attempts'][0]['cache_hit']
    assert budget.spent == 68 and len(cache) == 2
    assert sources == original


def test_real_callback_selects_only_frozen_witnesses_and_retains_paid_refits_after_failure(
        tmp_path, monkeypatch):
    _, agent, lab, question, _, callback, _ = paid_scene(tmp_path, mismatch=True)
    previous = deepcopy(question['native_condition_research'][0])
    target = callback['structural_discrimination']
    selected = structural_inquiry._frozen_structure_refit_inputs(question, target)
    assert len(selected) == 2
    assert {item['source_ref']['json_path'][-1] for item in selected} == {0, 1}
    real_assess = native._assess_matched_group
    monkeypatch.setattr(native, '_assess_matched_group',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('after-paid-refit')))
    with pytest.raises(RuntimeError, match='after-paid-refit'):
        structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    cache = question['native_structure_planning'][0]['refit_results']
    assert len(cache) == 2 and all(item['status'] == 'fitted' for item in cache)
    assert question['native_condition_research'][0] == previous
    retained = tmp_path / 'paid_refit_cache.json'
    write_json_checkpoint(retained, cache)
    question['native_structure_planning'][0]['refit_results'] = read_json_checkpoint(retained)
    monkeypatch.setattr(native, '_assess_matched_group', real_assess)
    monkeypatch.setattr(native, 'replay_research_program',
        lambda *a, **k: pytest.fail('interrupted later search lost its paid witness fits'))
    feedback = structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    result = question['native_condition_research'][-1]
    assert feedback['status'] != 'search_incomplete'
    assert result['retained_structure_refit']['complete']
    assert all(item['cache_hit'] for item in result['retained_structure_refit']['attempts'])
    stage = next(item for item in result['search_stages']
                 if item['stage'] == 'retained_structure_refit')
    assert stage['compute_cost'] == stage['completed_fits'] == 0
    costs = deepcopy(agent.costs)
    again = structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    assert again['cache_hit'] and agent.costs == costs

