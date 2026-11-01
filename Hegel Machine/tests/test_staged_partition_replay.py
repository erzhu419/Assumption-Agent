"""A next-stage condition consumes the actual completed cheap refit."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.research_drafts import _rebind_step
from hegel_machine.cognition.research_program_inputs import abstract_research_inputs
from hegel_machine.cognition.research_programs import (
    evaluate_research_step_revisions, quote_research_step_revisions,
    replay_research_program, research_program_replay_inputs,
)
from hegel_machine.cognition.staged_method_research import staged_program
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import quote_transfer_partition
from test_transfer_partitions import observations, proposal


def fixture(radius=.85):
    proposed = proposal()
    baseline = proposed['baseline_research_program']
    continuation = proposed['candidates'][0]['research_program']
    continuation['transfer_partition']['second_stage'] = {
        'operation': 'resolve_relation_intervals', 'compute_limit': 15000}
    guard = Program('ge', children=(Program('feature', name='public_0'),
                                    Program('const', value=1e-7)))
    body = staged_program(baseline, continuation, guard.to_dict(),
        [{'path': ['residual', 'loss'], 'kind': 'number'}], [{'file': 'paid/source_pair.json'}])
    rows = observations(radius=radius)
    return body, rows[:24], rows[24:]


@pytest.mark.parametrize('radius,selected', [(.4, 'baseline'), (.85, 'continuation')])
def test_ordinary_replay_only_reidentifies_after_a_complete_unexplained_refit(monkeypatch, radius, selected):
    import hegel_machine.cognition.research_programs as module
    body, training, feedback = fixture(radius)
    original = deepcopy(body)
    fit = module._Fitter.fit
    calls = []

    def record(self, terms):
        calls.append(tuple(terms))
        return fit(self, terms)

    monkeypatch.setattr(module._Fitter, 'fit', record)
    quote = quote_transfer_partition(body, training, feedback_rows=feedback)
    budget = SearchBudget(quote['required_compute'])
    entry = replay_research_program(body, training, budget, feedback_rows=feedback)
    result = entry['staged_research_result']
    assert result['execution_complete'] and result['selected_stage'] == selected
    assert result['compute_cost'] == budget.spent <= quote['required_compute']
    assert loss(entry['candidate'].program, feedback, 0., .5) < 1e-7
    assert result['baseline']['execution_complete']
    assert len(research_program_fit_contract(entry['research_program'])['fit_terms']) == 3
    assert body == original
    if selected == 'baseline':
        assert len(calls) == 1 and 'continuation' not in result
        assert result['guard']['status'] == 'evaluated' and not result['guard']['decision']
    else:
        assert len(calls) > 1 and result['guard']['decision'] is True
        assert result['guard']['profile']['residual']['loss'] == result['baseline']['all_paid_loss']
        assert result['continuation']['all_paid_loss'] < result['baseline']['all_paid_loss']


def test_guard_can_decline_and_missing_input_is_not_a_negative_method_result():
    body, training, feedback = fixture()
    body['staged_research']['condition'] = Program('const', value=0.).to_dict()
    entry = replay_research_program(body, training, SearchBudget(10000), feedback_rows=feedback)
    result = entry['staged_research_result']
    assert result['execution_complete'] and result['status'] == 'continuation_declined'
    assert 'continuation' not in result and result['baseline']['all_paid_loss'] > 1e-7
    body['staged_research']['condition'] = Program('feature', name='public_0').to_dict()
    body['staged_research']['coordinate_paths'] = [{'path': ['residual', 'unobserved'], 'kind': 'number'}]
    entry = replay_research_program(body, training, SearchBudget(10000), feedback_rows=feedback)
    assert entry['staged_research_result']['status'] == 'condition_inputs_unavailable'
    assert not entry['transfer_partition_search']['completed']


def test_unfunded_continuation_preserves_completed_p0_and_incomplete_status():
    body, training, feedback = fixture()
    first = quote_transfer_partition(body['staged_research']['baseline'], training)
    budget = SearchBudget(first['required_compute'] + len(feedback) + 1)
    entry = replay_research_program(body, training, budget, feedback_rows=feedback)
    result = entry['staged_research_result']
    assert result['status'] == 'continuation_incomplete' and result['budget_incomplete']
    assert not result['execution_complete'] and not entry['transfer_partition_search']['completed']
    assert result['selected_stage'] == 'baseline' and result['baseline']['execution_complete']
    assert result['compute_cost'] == budget.spent <= budget.limit


def test_missing_paid_feedback_does_not_authorize_a_stage_decision():
    body, training, _ = fixture()
    entry = replay_research_program(body, training, SearchBudget(10000))
    result = entry['staged_research_result']
    assert not result['execution_complete'] and result['status'] == 'baseline_incomplete'
    assert result['guard']['status'] == 'not_evaluated'
    assert 'continuation' not in result and result['baseline']['feedback_loss'] is None


def test_already_satisfied_result_stops_even_if_frozen_guard_predicts_continue():
    body, training, feedback = fixture(.4)
    body['staged_research']['condition'] = Program('const', value=1.).to_dict()
    entry = replay_research_program(body, training, SearchBudget(10000), feedback_rows=feedback)
    result = entry['staged_research_result']
    assert result['status'] == 'baseline_supported' and 'continuation' not in result
    assert result['guard']['decision'] and result['guard']['ignored_because_goal_satisfied']


def test_completed_worse_continuation_cannot_replace_p0(monkeypatch):
    import hegel_machine.cognition.transfer_partitions as module
    body, training, feedback = fixture()
    native = module.replay_transfer_partition

    def worse(payload, *args, **kwargs):
        entry = native(payload, *args, **kwargs)
        if payload['transfer_partition']['kind'] == 'Pi1':
            entry['candidate'] = replace(entry['candidate'], program=Program('const', value=100.))
            entry['training_loss'] = 10000.
            entry['transfer_partition_followup'].update(feedback_loss=10000.)
        return entry

    monkeypatch.setattr(module, 'replay_transfer_partition', worse)
    result = replay_research_program(body, training, SearchBudget(100000), feedback_rows=feedback)
    evidence = result['staged_research_result']
    assert evidence['status'] == 'baseline_retained_after_comparison'
    assert evidence['execution_complete'] and evidence['selected_stage'] == 'baseline'
    assert result['candidate'].program.to_dict() == evidence['baseline']['program']


def paid_fixture():
    body, training, feedback = fixture()
    records = {}
    for name in ('baseline', 'continuation'):
        part = body['staged_research'][name]
        entry = replay_research_program(part, training, SearchBudget(100000), feedback_rows=feedback)
        records[name] = {'replay_inputs': research_program_replay_inputs(part),
            'training': training, 'feedback': feedback, 'risk_quantile': .5, 'tolerance': 1e-7,
            'entry': entry, 'execution_complete': True,
            'feedback_loss': loss(entry['candidate'].program, feedback, 0., .5),
            'cost_ref': {'file': f'paid/{name}.json'}}
    return body, training, feedback, records


def test_paid_stages_execute_new_guard_without_repeating_fits_or_predictions(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    import hegel_machine.cognition.evaluation as evaluation
    body, training, feedback, records = paid_fixture()
    saved = deepcopy(records)
    monkeypatch.setattr(module._Fitter, 'fit', lambda *a: pytest.fail('repeated paid fit'))
    monkeypatch.setattr(evaluation, 'loss', lambda *a: pytest.fail('repeated paid score'))
    budget = SearchBudget(1)
    entry = replay_research_program(body, training, budget, feedback_rows=feedback, paid_stages=records)
    result = entry['staged_research_result']
    assert result['execution_complete'] and result['selected_stage'] == 'continuation'
    assert result['compute_cost'] == budget.spent == 1
    assert all(result[name]['reused'] for name in ('baseline', 'continuation'))
    assert records == saved


def test_competing_conditions_do_not_silently_use_first_or_repeat_paid_research(monkeypatch):
    import hegel_machine.cognition.transfer_partitions as module
    body, training, feedback, records = paid_fixture()
    first = body['staged_research']['condition']
    body['staged_research']['conditions'] = [first, Program('const', value=0.).to_dict()]
    monkeypatch.setattr(module, 'replay_transfer_partition', lambda *a, **kw:
                        pytest.fail('repeated a completed stage'))
    budget = SearchBudget(2)
    entry = replay_research_program(body, training, budget, feedback_rows=feedback, paid_stages=records)
    result = entry['staged_research_result']
    assert result['status'] == 'competing_conditions_unresolved'
    assert not result['execution_complete'] and not entry['transfer_partition_search']['completed']
    assert result['guard']['decision'] is None and 'continuation' not in result
    assert budget.spent == 2 and result['baseline']['reused']


def test_surviving_condition_consensus_reuses_paid_continuation_and_changes_identity(monkeypatch):
    import hegel_machine.cognition.transfer_partitions as module
    body, training, feedback, records = paid_fixture()
    key = research_program_replay_inputs(body)
    first = body['staged_research']['condition']
    body['staged_research']['conditions'] = [first, Program('const', value=1.).to_dict()]
    assert research_program_replay_inputs(body) != key
    quote = quote_transfer_partition(body, training, feedback_rows=feedback)
    assert quote['guard_compute'] == 2
    monkeypatch.setattr(module, 'replay_transfer_partition', lambda *a, **kw:
                        pytest.fail('repeated a completed stage'))
    budget = SearchBudget(2)
    entry = replay_research_program(body, training, budget, feedback_rows=feedback, paid_stages=records)
    result = entry['staged_research_result']
    assert result['execution_complete'] and result['selected_stage'] == 'continuation'
    assert len(result['guard']['candidate_predictions']) == 2
    assert result['continuation']['reused'] and budget.spent == 2
    rebound = _rebind_step(body, {'public_0': 'not_a_guard_coordinate'})
    assert rebound['staged_research']['conditions'] == body['staged_research']['conditions']


def test_unfunded_condition_set_is_not_a_single_condition_vote():
    body, training, feedback, records = paid_fixture()
    body['staged_research']['conditions'] = [body['staged_research']['condition'],
                                           Program('const', value=1.).to_dict()]
    entry = replay_research_program(body, training, SearchBudget(1), feedback_rows=feedback, paid_stages=records)
    result = entry['staged_research_result']
    assert result['status'] == 'condition_compute_unresolved' and result['budget_incomplete']
    assert not result['execution_complete'] and 'continuation' not in result


@pytest.mark.parametrize('changed', ['rows', 'body', 'tolerance'])
def test_paid_stage_reuse_requires_same_evidence_and_executed_body(changed):
    body, training, feedback, records = paid_fixture()
    if changed == 'rows':
        records['baseline']['training'] = [replace(training[0], target=training[0].target + 1), *training[1:]]
    elif changed == 'body':
        records['baseline']['replay_inputs']['transfer_partition']['max_attempts'] = 2
    else:
        records['baseline']['tolerance'] = .01
    entry = replay_research_program(body, training, SearchBudget(100000), feedback_rows=feedback, paid_stages=records)
    assert not entry['staged_research_result']['baseline']['reused']
    assert entry['staged_research_result']['baseline']['compute_cost'] > 0


def test_rebinding_and_replay_identity_include_both_stages_but_not_guard_sensor_names():
    body, training, feedback = fixture()
    mapping = {name: f'new_{name}' for name in training[0].features}
    mapping['public_0'] = 'this_is_not_a_guard_coordinate'
    rebound = _rebind_step(body, mapping)
    assert rebound['staged_research']['condition'] == body['staged_research']['condition']
    renamed = lambda row: replace(row, features={mapping[key]: value for key, value in row.features.items()})
    entry = replay_research_program(body, list(map(renamed, training)), SearchBudget(100000),
        feedback_rows=list(map(renamed, feedback)), feature_mapping=mapping)
    assert entry['staged_research_result']['execution_complete']
    assert all(name.startswith('new_') for name in entry['candidate'].program.required_features())
    key = research_program_replay_inputs(body)
    changed = deepcopy(body)
    changed['staged_research']['evidence_refs'] = [{'file': 'same_evidence_moved.json'}]
    assert research_program_replay_inputs(changed) == key
    changed['staged_research']['continuation']['transfer_partition']['max_attempts'] = 3
    assert research_program_replay_inputs(changed) != key
    changed = deepcopy(body)
    changed['staged_research']['condition'] = Program('const', value=0.).to_dict()
    assert research_program_replay_inputs(changed) != key


def test_abstraction_recovers_nested_scientific_stages_without_sensorizing_guard():
    body, training, _ = fixture()
    fields = {name: name for name in training[0].features}
    schema = {'objects': ['object'], 'fields': {'object': fields},
              'members': {'object': ['object']}, 'controls': {'object': {}}, 'focus_control': None}
    spec = TaskSpec('sensor_view', 'Measured current inputs', tuple(fields), (), (),
                    context={'object_schema': json.dumps(schema)})
    result = abstract_research_inputs(body, spec, KnowledgeLibrary())
    assert result is not None
    assert result['staged_research']['condition'] == body['staged_research']['condition']
    assert 'public_0' not in result['input_features']
    assert set(fields) == set(result['input_features'])
    assert research_program_replay_inputs(result) == research_program_replay_inputs(body)


def test_ordinary_paired_replay_quotes_and_consumes_the_stage_completion():
    body, training, feedback = fixture()
    prepared = {'status': 'proposed', 'baseline_research_program': body['staged_research']['baseline'],
        'candidates': [{'research_program': body, 'edit': {'operation': 'refit_then_reidentify'}}]}
    quote = quote_research_step_revisions(prepared['baseline_research_program'], training, feedback,
        prepared_proposal=prepared)
    assert quote['status'] == 'quoted'
    result = evaluate_research_step_revisions(prepared['baseline_research_program'], training, feedback,
        SearchBudget(2 * quote['per_method_compute_upper']),
        per_method_compute=quote['per_method_compute_upper'], prepared_proposal=prepared)
    assert result['methods'][1]['execution_complete']
    assert result['methods'][1]['assessment_role'] == 'adaptive_development'
    assert result['methods'][1]['validation_loss'] < 1e-7


def test_native_research_invocation_keeps_feedback_and_four_draft_limit(monkeypatch):
    from types import SimpleNamespace
    from hegel_machine.cognition import research_programs, transfer_partitions
    from hegel_machine.cognition.models import Candidate
    from test_native_decision_checkpoint import prepared_scene

    agent, frame, _ = prepared_scene()
    body, _, _ = fixture()
    template = SimpleNamespace(roles=('measured',), source_features=('drive',))
    plans = [('source', body, template, [{'measured': 'x'}])]
    quotes, calls = [], []

    def quote(bound, training, *, feedback_rows):
        quotes.append(feedback_rows)
        assert bound['staged_research']['condition'] == body['staged_research']['condition']
        return {'required_compute': 24000}

    def replay(payload, training, budget, **kwargs):
        calls.append(kwargs)
        assert budget.spend(1)
        return {'candidate': Candidate('returned', Program('const', value=0.), 'test'),
                'training_loss': 0., 'kind': 'replay'}

    monkeypatch.setattr(transfer_partitions, 'quote_transfer_partition', quote)
    monkeypatch.setattr(research_programs, 'replay_research_program', replay)
    assert agent._replay_search_allowance(plans, frame['training'], frame['comparison'], 30000) == 24000
    assert quotes == [frame['comparison']]
    claims = agent._replay_program_claims(frame['spec'], frame['training'], SearchBudget(10), [],
        prepared=plans, feedback_rows=frame['comparison'])
    assert len(claims) == len(calls) == 1 <= 4
    assert calls[0]['feedback_rows'] is frame['comparison']
    assert calls[0]['tolerance'] == agent.config.minimum_improvement
