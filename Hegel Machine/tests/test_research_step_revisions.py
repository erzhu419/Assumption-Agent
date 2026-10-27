"""Actual one-step replay interventions, with matched fitting and assessment."""

from copy import deepcopy
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.research_programs import (
    _plan, evaluate_research_step_revisions, propose_research_step_revisions,
    quote_research_step_revisions, research_program_replay_inputs)


REFERENCES = [{'file': '/immutable/paid_view.json', 'json_path': ['training']},
              {'file': '/immutable/paid_view.json', 'json_path': ['validation']}]


def recipe(*, conditional=False):
    term = Program('feature', name='u')
    step = {'operation': 'fit_component', 'question': 'Fit the measured input',
        'program': term, 'loss': 1.}
    if conditional:
        step.update(operation='discover_conditional_coordinates', gate_coordinate=term.to_dict(),
            branch_basis=[term.to_dict()], threshold=0., width=1, placement='inside_window')
    return _plan({'fit_terms': (term,), 'coefficients': [0., 1.],
        'program': term, 'loss': 1.}, [step], 24, 100)


def rows(seed, mode='square', *, lengths=(12, 12)):
    rng, result = Random(seed), []
    for length in lengths:
        history = []
        for _ in range(length):
            value = rng.uniform(-2., 2.)
            target = value * value if mode == 'square' else (
                .3 * value + 1.4 * (history[-1]['u'] if history else 0.))
            result.append(Observation({'u': value}, target, tuple(history)))
            history.append({'u': value})
    return result


def compare(payload=None, *, mode='square', previous=None, refs=REFERENCES, allowance=1000,
            max_candidates=8, budget=100000):
    return evaluate_research_step_revisions(payload or recipe(), rows(7, mode), rows(31, mode),
        SearchBudget(budget), per_method_compute=allowance, source_refs=refs,
        max_candidates=max_candidates, previous=previous)


def test_generation_changes_real_replay_inputs_and_uses_public_guarded_history():
    original = recipe()
    unchanged = deepcopy(original)
    result = propose_research_step_revisions(original, rows(7), max_candidates=8)
    baseline = research_program_replay_inputs(original)
    assert result['candidates']
    assert result['observed_history_bound'] == 11
    assert any(Program.from_dict(item['edit']['term']).history_depth
        for item in result['candidates'] if item['edit'].get('term'))
    for item in result['candidates']:
        assert item['replay_inputs'] != baseline
        assert item['replay_inputs'] == research_program_replay_inputs(item['research_program'])
        assert item['status'] == 'unmeasured_step_intervention'
        assert 'case_id' not in item['edit']
    assert original == unchanged


@pytest.mark.parametrize('mode', ['square', 'history'])
def test_complete_same_material_pairs_select_different_executable_steps_from_actual_losses(mode):
    result = compare(mode=mode)
    assert result['status'] == 'development_step_revision_selected'
    selected = result['selected_revision']
    assert selected and selected['validation_loss'] < 1e-10
    term = Program.from_dict(selected['edit']['term'])
    assert bool(term.history_depth) is (mode == 'history')
    assert selected['replayed_research_program']['graph']['nodes'][
        selected['replayed_research_program']['output_node']]['answer'] == selected['paired_program']
    assert all(item['execution_complete'] and item['same_training_and_validation']
        and item['equal_method_compute_ceiling'] == 1000 for item in result['pairs'])
    assert result['compute_cost'] <= result['required_compute']
    assert not result['independent_support']
    assert result['scope'] == 'same_paid_split_development_step_intervention'


def test_equal_compute_window_and_split_do_not_mean_equal_actual_cost_or_independent_confirmation():
    result = compare()
    assert len({item['new_compute_cost'] for item in result['methods']}) > 1
    assert all(item['new_compute_cost'] <= 1000 for item in result['methods'])
    assert all(pair['scope'] == result['scope'] for pair in result['pairs'])
    assert result['basis']['training_rows'] == result['basis']['validation_rows'] == 24


def test_insufficient_complete_budget_starts_no_baseline_or_candidate_fit(monkeypatch):
    import hegel_machine.cognition.research_programs as module

    def unexpected(*args, **kwargs):
        raise AssertionError('unquoted scientific work started')

    monkeypatch.setattr(module, 'replay_research_program', unexpected)
    result = compare(budget=1)
    assert result['status'] == 'insufficient_budget'
    assert result['budget_incomplete'] and result['compute_cost'] == 0
    assert result['denied_request']['required_compute'] == result['required_compute']


def test_completed_paid_replays_and_assessments_are_reused_without_recomputation(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    import hegel_machine.cognition.evaluation as evaluation

    completed = compare()
    original = deepcopy(completed)

    def unexpected(*args, **kwargs):
        raise AssertionError('complete paid replay or validation repeated')

    monkeypatch.setattr(module, 'replay_research_program', unexpected)
    monkeypatch.setattr(evaluation, 'loss', unexpected)
    reused = compare(previous=completed, budget=0)
    assert reused['compute_cost'] == reused['required_compute'] == 0
    assert reused['paid_compute_cost'] == completed['paid_compute_cost']
    assert reused['selected_revision'] == completed['selected_revision']
    assert all(item['reused'] and item['new_compute_cost'] == 0 for item in reused['methods'])
    assert completed == original


def test_changed_paid_view_or_actual_method_inputs_invalidate_replay_reuse():
    first = compare()
    changed = compare(previous=first, refs=[{'file': '/immutable/new_paid_view.json'}])
    assert changed['compute_cost'] > 0 and not any(item['reused'] for item in changed['methods'])
    new_method = recipe()
    node = new_method['term_nodes'][0]
    new_method['graph']['nodes'][node]['answer'] = Program('mul', children=(
        Program('feature', name='u'), Program('feature', name='u'))).to_dict()
    after = compare(new_method, previous=first)
    assert after['compute_cost'] > 0 and not any(item['reused'] for item in after['methods'])


def test_incomplete_fits_keep_costs_without_creating_a_negative_step_result():
    result = compare(allowance=25)
    assert result['status'] == 'incomplete_step_comparison'
    assert result['budget_incomplete']
    assert all(item['status'] == 'incomplete_not_negative' for item in result['pairs'])
    assert result['selected_revision'] is None
    assert result['paid_compute_cost'] == sum(item['paid_compute_cost'] for item in result['methods'])


def test_method_exceptions_retain_paid_work_and_are_not_negative_examples(monkeypatch):
    import hegel_machine.cognition.research_programs as module

    def failed(payload, training, budget, **kwargs):
        budget.spend(7)
        raise ValueError('measured replay cannot complete')

    monkeypatch.setattr(module, 'replay_research_program', failed)
    result = compare()
    assert result['status'] == 'incomplete_step_comparison'
    assert all(item['status'] == 'incomplete_not_negative' for item in result['pairs'])
    assert all(item['execution_error']['type'] == 'ValueError' for item in result['methods'])
    assert result['paid_compute_cost'] == 7 * len(result['methods'])


def test_public_role_mapping_is_applied_once_before_creating_target_history_coordinates():
    mapped = [Observation({'control': row.features['u']}, row.target,
        tuple({'control': past['u']} for past in row.history)) for row in rows(7, 'history')]
    proposed = propose_research_step_revisions(recipe(), mapped,
        feature_names=('control',), feature_mapping={'u': 'control'})
    assert proposed['candidates']
    for item in proposed['candidates']:
        for value in item['replay_inputs']['retained_terms']:
            assert Program.from_dict(value).required_features() <= {'control'}


def test_historical_trace_prose_is_not_an_executable_step_intervention():
    first, changed = recipe(), recipe()
    changed['selection_steps'][0]['question'] = 'A new scientific sounding instruction'
    changed['selection_steps'][0]['loss'] = 0.
    left = propose_research_step_revisions(first, rows(7))
    right = propose_research_step_revisions(changed, rows(7))
    assert left['baseline_replay_inputs'] == right['baseline_replay_inputs']
    assert [item['replay_inputs'] for item in left['candidates']] == [
        item['replay_inputs'] for item in right['candidates']]


def test_unadapted_specialized_step_contract_does_not_trigger_a_generic_replay(monkeypatch):
    import hegel_machine.cognition.research_programs as module

    original = recipe()
    original['selection_steps'][0]['operation'] = 'resolve_relation_intervals'
    monkeypatch.setattr(module, 'replay_research_program', lambda *a, **kw: pytest.fail('wrong contract'))
    result = compare(original)
    assert result['status'] == 'step_contract_not_established'
    assert result['compute_cost'] == result['required_compute'] == 0


def test_roundoff_differences_below_original_quality_contract_do_not_force_a_new_step():
    data = [Observation(row.features, row.features['u'], row.history) for row in rows(7)]
    held = [Observation(row.features, row.features['u'], row.history) for row in rows(31)]
    result = evaluate_research_step_revisions(recipe(), data, held, SearchBudget(100000),
        per_method_compute=1000, source_refs=REFERENCES)
    assert result['methods'][0]['validation_loss'] < 1e-10
    assert result['status'] == 'complete_no_step_improvement'
    assert result['selected_revision'] is None


def test_equal_quality_cheaper_complete_step_uses_the_shared_process_return():
    original = recipe()
    extra = Program('mul', children=(Program('feature', name='u'), Program('feature', name='u')))
    original['graph']['nodes']['extra'] = deepcopy(original['graph']['nodes'][original['term_nodes'][0]])
    original['graph']['nodes']['extra']['answer'] = extra.to_dict()
    original['term_nodes'].append('extra')
    data = [Observation(row.features, row.features['u'], row.history) for row in rows(7)]
    held = [Observation(row.features, row.features['u'], row.history) for row in rows(31)]
    result = evaluate_research_step_revisions(original, data, held, SearchBudget(100000),
        per_method_compute=1000, source_refs=REFERENCES, max_candidates=16)
    cheaper = [pair for pair in result['pairs'] if pair['process_return']['helpful'] is True]
    assert cheaper and any(pair['edit']['kind'] == 'remove_retained_input' for pair in cheaper)
    assert result['selected_revision']
    assert result['selected_revision']['validation_loss'] < 1e-10


def test_conditional_branch_revision_is_actually_consumed_by_the_existing_specialized_replayer():
    training = [Observation(row.features, row.target) for row in rows(7)]
    validation = [Observation(row.features, row.target) for row in rows(31)]
    result = evaluate_research_step_revisions(recipe(conditional=True), training, validation,
        SearchBudget(40000), per_method_compute=10000, source_refs=REFERENCES, max_candidates=3)
    branch = next(pair for pair in result['pairs'] if pair['edit']['kind'] == 'add_branch_coordinate')
    assert branch['execution_complete'] and branch['status'] == 'local_step_improvement'
    fitted = result['methods'][branch['method_index']]
    assert fitted['validation_loss'] < 1e-10
    expected = branch['edit']['term']
    # The ordinary conditional search may find the condition unnecessary.
    # Its returned executable dependencies must still contain the new basis.
    assert any(fitted['research_program']['graph']['nodes'][key]['answer'] == expected
        for key in fitted['research_program']['term_nodes'])
    assert fitted['program'] == expected


def test_full_quote_counts_actual_gate_brackets_and_does_not_fit_or_change_the_recipe(monkeypatch):
    import hegel_machine.cognition.research_programs as module

    original = recipe(conditional=True)
    frozen = deepcopy(original)
    monkeypatch.setattr(module._Fitter, 'fit', lambda *a, **kw: pytest.fail('fit during quote'))
    quoted = quote_research_step_revisions(original, rows(7), rows(31), max_candidates=3)
    assert quoted['status'] == 'quoted'
    assert quoted['method_count'] == 4
    assert quoted['complete_pass_compute_upper'] == 4 * quoted['per_method_compute_upper']
    assert all(gate['brackets'] == 19 for method in quoted['methods'] for gate in method['gate_brackets'])
    assert quoted['coordinate_readout_compute'] == 2 * 24  # u and u*u, shared across methods
    assert quoted['new_fits'] == quoted['new_world_predictions'] == 0
    assert original == frozen


def test_complete_finite_quote_covers_retained_plain_parameter_and_sparse_fit_work():
    original, training, validation = recipe(conditional=True), rows(7), rows(31)
    quoted = quote_research_step_revisions(original, training, validation, max_candidates=3)
    allowance = quoted['per_method_compute_upper']
    result = evaluate_research_step_revisions(original, training, validation,
        SearchBudget(quoted['complete_pass_compute_upper']), per_method_compute=allowance,
        source_refs=REFERENCES, max_candidates=3)
    assert not result['budget_incomplete']
    assert all(method['execution_complete'] for method in result['methods'])
    assert all(method['new_compute_cost'] <= quoted_method['compute_upper']
        for method, quoted_method in zip(result['methods'], quoted['methods']))
    assert result['compute_cost'] <= quoted['complete_pass_compute_upper']


def test_screened_or_unadapted_specialized_full_quote_remains_unknown():
    result = quote_research_step_revisions(recipe(conditional=True), rows(7)[:6], rows(31),
        max_candidates=1)
    assert result['status'] == 'UNKNOWN'
    assert result['reason'] == 'screened_conditional_fit_complete_bound_not_established'
    assert result['per_method_compute_upper'] is None


def paid_baseline(first, *, scored):
    old = first['methods'][0]
    result = {'basis': deepcopy(first['baseline_basis']), 'execution_complete': True,
        'program': deepcopy(old['program']), 'research_program': deepcopy(old['research_program']),
        'training_loss': old['training_loss'], 'fit_compute_cost': old['paid_compute_cost'] - 24,
        'cost_ref': {'file': '/immutable/old_seed_account.json'}}
    if scored:
        result.update(validation_loss=old['validation_loss'], paid_compute_cost=old['paid_compute_cost'])
    return result


@pytest.mark.parametrize('scored', [False, True])
def test_exact_paid_seed_is_not_refitted_and_only_missing_holdout_assessment_is_charged(monkeypatch, scored):
    import hegel_machine.cognition.research_programs as module

    first = compare()
    old = paid_baseline(first, scored=scored)
    actual = module.replay_research_program

    def replay(payload, *args, **kwargs):
        assert research_program_replay_inputs(payload) != old['basis']['replay_inputs']
        return actual(payload, *args, **kwargs)

    monkeypatch.setattr(module, 'replay_research_program', replay)
    result = evaluate_research_step_revisions(recipe(), rows(7), rows(31), SearchBudget(100000),
        per_method_compute=1000, source_refs=REFERENCES, paid_baseline=old)
    baseline = result['methods'][0]
    assert baseline['reused'] and baseline['new_compute_cost'] == (0 if scored else 24)
    assert baseline['cost_ref'] == old['cost_ref']
    assert baseline['paid_compute_cost'] == first['methods'][0]['paid_compute_cost']
    assert baseline['validation_loss'] == first['methods'][0]['validation_loss']
    assert result['compute_cost'] == sum(method['new_compute_cost'] for method in result['methods'])
    assert result['required_compute'] == 8 * 1000 + (0 if scored else 24)


@pytest.mark.parametrize('change', ['unknown_seed', 'paid_view', 'split', 'method_input'])
def test_unmatched_paid_seed_is_unresolved_instead_of_silently_repeating_the_old_fit(monkeypatch, change):
    import hegel_machine.cognition.research_programs as module

    first = compare()
    old = paid_baseline(first, scored=False)
    if change == 'unknown_seed':
        old = {}
    elif change == 'paid_view':
        old['basis']['source_refs'] = [{'file': '/immutable/other_paid_view.json'}]
    elif change == 'split':
        old['basis']['validation_rows'] = 12
    else:
        old['basis']['replay_inputs']['retained_terms'] = []
    monkeypatch.setattr(module, 'replay_research_program', lambda *a, **kw: pytest.fail('old paid fit repeated'))
    result = evaluate_research_step_revisions(recipe(), rows(7), rows(31), SearchBudget(100000),
        per_method_compute=1000, source_refs=REFERENCES, paid_baseline=old)
    assert result['status'] == 'paid_baseline_not_reusable'
    assert result['compute_cost'] == 0 and not result['methods']


def test_shared_python_deadline_stops_the_whole_matrix_and_preserves_current_slot_cost(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    from hegel_machine.cognition.experiment_runtime import PythonExecutionDeadline

    calls = []

    def timeout(payload, training, budget, **kwargs):
        calls.append(payload)
        budget.spend(7)
        raise PythonExecutionDeadline('the registered stage window ended')

    monkeypatch.setattr(module, 'replay_research_program', timeout)
    result = compare()
    assert len(calls) == len(result['methods']) == 1
    assert result['time_incomplete'] and result['status'] == 'incomplete_step_comparison'
    assert result['methods'][0]['execution_error']['type'] == 'PythonExecutionDeadline'
    assert result['compute_cost'] == result['paid_compute_cost'] == 7
    assert result['unexecuted_method_indices'] == list(range(1, 9))
    assert not result['pairs'] and result['selected_revision'] is None


def test_deadline_during_missing_paid_holdout_stops_before_any_new_fit(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    import hegel_machine.cognition.evaluation as evaluation
    from hegel_machine.cognition.experiment_runtime import PythonExecutionDeadline

    previous = compare()
    old = paid_baseline(previous, scored=False)
    monkeypatch.setattr(module, 'replay_research_program', lambda *a, **kw: pytest.fail('new slot after deadline'))

    def timeout(*args, **kwargs):
        raise PythonExecutionDeadline('the registered stage window ended')

    monkeypatch.setattr(evaluation, 'loss', timeout)
    result = evaluate_research_step_revisions(recipe(), rows(7), rows(31), SearchBudget(100000),
        per_method_compute=1000, source_refs=REFERENCES, paid_baseline=old)
    assert result['time_incomplete'] and result['status'] == 'incomplete_paid_baseline_assessment'
    assert result['compute_cost'] == 24 and len(result['methods']) == 1
    assert result['methods'][0]['cost_ref'] == old['cost_ref']
    assert result['selected_revision'] is None


def test_slot_progress_survives_interruption_and_reuses_the_paid_fit_and_assessment(monkeypatch):
    import hegel_machine.cognition.research_programs as module

    checkpoints, original_replay = [], module.replay_research_program
    first_budget = SearchBudget(100000)

    def interrupted_checkpoint(partial):
        checkpoints.append(partial)
        raise RuntimeError('interrupt after the completed slot was persisted')

    with pytest.raises(RuntimeError, match='completed slot was persisted'):
        evaluate_research_step_revisions(recipe(), rows(7), rows(31), first_budget,
            per_method_compute=1000, source_refs=REFERENCES, on_progress=interrupted_checkpoint)
    saved, = checkpoints
    assert len(saved['methods']) == 1 and saved['methods'][0]['execution_complete']
    assert saved['compute_cost'] == saved['paid_compute_cost'] == first_budget.spent > 0
    assert saved['unexecuted_method_indices'] == list(range(1, 9))
    frozen = deepcopy(saved)
    new_checkpoints = []

    def remaining_only(payload, *args, **kwargs):
        assert research_program_replay_inputs(payload) != saved['basis']['replay_inputs'][0]
        return original_replay(payload, *args, **kwargs)

    monkeypatch.setattr(module, 'replay_research_program', remaining_only)
    result = evaluate_research_step_revisions(recipe(), rows(7), rows(31), SearchBudget(100000),
        per_method_compute=1000, source_refs=REFERENCES, previous=saved,
        on_progress=new_checkpoints.append)
    assert result['methods'][0]['reused'] and result['methods'][0]['new_compute_cost'] == 0
    assert len(new_checkpoints) == 8
    assert len(new_checkpoints[0]['methods']) == 2  # retained callback values stay immutable
    assert new_checkpoints[-1]['paid_compute_cost'] == result['paid_compute_cost']
    assert result['paid_compute_cost'] == saved['paid_compute_cost'] + result['compute_cost']
    assert result['selected_revision'] and saved == frozen
