"""Sequential Pi1 research consumes the local research's actual fitted parent."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.research_programs import replay_research_program
from hegel_machine.cognition.staged_method_research import staged_program
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import quote_transfer_partition
from test_parent_relation_revisions import scene
from test_transfer_partitions import plan


def fixture(kind='Pi3', radius=.85):
    parent, source_rows = scene(source_direction='right' if kind == 'Pi2' else None, count=64)
    baseline = deepcopy(parent['research_program'])
    source_terms = research_program_fit_contract(baseline)['fit_terms']
    source_symbolic = [Program.from_dict(term) for term in baseline['selection_steps'][0]['parameter_terms']]

    def redirect(node, direction):
        return replace(node, name=direction) if node.op == 'neighbor_sum' else replace(
            node, children=tuple(redirect(child, direction) for child in node.children))

    alternatives = []
    for option in range(2):
        def revise(term):
            return (redirect(term, ('right', 'left')[option]) if kind == 'Pi3' else
                    term.remap_features({'x0': 'p0', 'x1': 'p1'} if option == 0 else
                                        {'x0': 'p1', 'x1': 'p0'}))
        alternatives.append({'fit_terms': [revise(term).to_dict() for term in source_terms],
                             'parameter_terms': [revise(term).to_dict() for term in source_symbolic]})
    baseline['transfer_partition'] = {'kind': kind, 'max_attempts': 2, 'max_candidates': 4,
        'released_groups': [], 'alternatives': alternatives,
        'search_domain': 'observed_local_revision_existing_vocabulary'}
    continuation = plan()
    continuation['transfer_partition'] = {'kind': 'Pi1', 'released_groups': ['r'],
        'max_attempts': 48, 'max_candidates': 4, 'search_domain': 'current_paid_geometry'}
    guard = Program('ge', children=(Program('feature', name='public_0'), Program('const', value=1e-7)))
    body = staged_program(baseline, continuation, guard.to_dict(),
        [{'path': ['residual', 'loss'], 'kind': 'number'}], [{'file': 'paid_local_research.json'}])
    body['staged_research']['operation'] = 'revise_then_reidentify'

    def reradius(node):
        return replace(node, children=(Program('const', value=radius), *node.children[1:])) \
            if node.op == 'neighbor_sum' else replace(node, children=tuple(reradius(child) for child in node.children))

    actual_terms = [reradius(Program.from_dict(term)) for term in alternatives[0]['fit_terms']]
    actual = _sum_terms(actual_terms, [2., -1., .5, 0., 0., 0., 3., -2.], .5)
    rows, history = [], []
    for index, original in enumerate(source_rows):
        fields = {**original.features, 'x1': .1 + (-1 if index % 2 else 1) *
                  (.2, .6, 1.1, .3, .7, .9, .5)[index % 7]}
        if kind == 'Pi2':
            fields.update(p0=fields['x0'], p1=fields['x1'], x1=3.)
        observed = Observation(fields, 0., tuple(history[-2:]))
        rows.append(replace(observed, target=actual.predict(observed)))
        history.append(fields)
    return body, rows[:40], rows[40:]


@pytest.mark.parametrize('kind', ['Pi2', 'Pi3'])
def test_continuation_uses_actual_eight_column_parent_and_paid_initializer(kind, monkeypatch):
    import hegel_machine.cognition.transfer_partitions as module
    body, training, feedback = fixture(kind)
    original, calls = deepcopy(body), []
    native = module.replay_transfer_partition

    def record(payload, *args, **kwargs):
        if payload['transfer_partition']['kind'] == 'Pi1':
            fit = research_program_fit_contract(payload)
            paid = kwargs['paid_initial']
            assert len(fit['fit_terms']) == 8
            assert fit['program'] == paid['entry']['candidate'].program
            assert fit == research_program_fit_contract(paid['entry']['research_program'])
            assert paid['training'] == training
            calls.append(payload)
        return native(payload, *args, **kwargs)

    monkeypatch.setattr(module, 'replay_transfer_partition', record)
    quote = quote_transfer_partition(body, training, feedback_rows=feedback)
    budget = SearchBudget(quote['required_compute'])
    entry = replay_research_program(body, training, budget, feedback_rows=feedback)
    result = entry['staged_research_result']
    assert len(calls) == 1 and result['execution_complete']
    assert result['selected_stage'] == 'continuation'
    assert result['baseline']['all_paid_loss'] > 1e-7
    assert result['continuation']['all_paid_loss'] < 1e-7
    assert result['guard']['profile']['residual']['loss'] == result['baseline']['all_paid_loss']
    assert result['parent_binding']['new_parent_binding']
    assert result['continuation']['parent_stage'] == 'baseline'
    assert result['continuation']['parent_parameter_revision']['new_parent_binding']
    assert result['continuation']['replay_inputs']['retained_terms'] == [
        term.to_dict() for term in research_program_fit_contract(calls[0])['fit_terms']]
    assert result['risk_quantile'] == .5 and result['tolerance'] == 1e-7
    assert entry['transfer_partition_search']['reused_initial_fits'] == 1
    assert len(research_program_fit_contract(entry['research_program'])['fit_terms']) == 8
    assert entry['research_program']['staged_research'] == original['staged_research']
    assert body == original and budget.spent <= quote['required_compute']
    assert all(item['fit_columns'] == 8 for item in quote['stages']['continuation']['parent_alternatives'])


def test_parent_quote_covers_every_legal_alternative_and_interval_followup():
    body, training, feedback = fixture()
    stage = body['staged_research']
    stage['continuation']['transfer_partition']['second_stage'] = {
        'operation': 'resolve_relation_intervals', 'compute_limit': 15000}
    alternative = stage['baseline']['transfer_partition']['alternatives'][1]
    extra = Program('mul', children=(Program('feature', name='z0'), Program('feature', name='z1'))).to_dict()
    alternative['fit_terms'].append(extra)
    alternative['parameter_terms'].append(extra)
    quote = quote_transfer_partition(body, training, feedback_rows=feedback)
    bound = quote['stages']['continuation']
    assert [item['fit_columns'] for item in bound['parent_alternatives']] == [8, 9]
    assert all(item['followup_compute_upper'] == 15000 for item in bound['parent_alternatives'])
    assert bound['required_compute'] == max(item['required_compute'] for item in bound['parent_alternatives'])
    assert quote['stages']['baseline']['required_compute'] == ((8 + 2) + (9 + 2)) * len(training)
    assert bound['required_compute'] > quote_transfer_partition(
        stage['continuation'], training, feedback_rows=feedback)['required_compute']


def test_missing_actual_parent_binding_returns_unresolved_without_using_source_pi1():
    body, training, feedback = fixture()
    for alternative in body['staged_research']['baseline']['transfer_partition']['alternatives']:
        alternative.pop('parameter_terms')
    quote = quote_transfer_partition(body, training, feedback_rows=feedback)
    assert quote['required_compute'] is None
    entry = replay_research_program(body, training, SearchBudget(100000), feedback_rows=feedback)
    result = entry['staged_research_result']
    assert result['status'] == 'continuation_dependency_unresolved'
    assert result['baseline']['execution_complete'] and not result['execution_complete']
    assert 'continuation' not in result and result['selected_stage'] == 'baseline'


def test_new_task_rebinds_again_and_guard_conflict_keeps_completed_local_result():
    body, training, feedback = fixture()
    first = replay_research_program(body, training, SearchBudget(100000), feedback_rows=feedback)
    _, new_training, new_feedback = fixture(radius=.65)
    second = replay_research_program(first['research_program'], new_training, SearchBudget(100000),
                                     feedback_rows=new_feedback)
    assert second['staged_research_result']['execution_complete']
    assert second['staged_research_result']['all_paid_loss'] < 1e-7
    assert second['candidate'].program != first['candidate'].program
    body['staged_research']['conditions'] = [body['staged_research']['condition'],
                                           Program('const', value=0.).to_dict()]
    conflict = replay_research_program(body, training, SearchBudget(100000), feedback_rows=feedback)
    result = conflict['staged_research_result']
    assert result['status'] == 'competing_conditions_unresolved'
    assert not result['execution_complete'] and 'continuation' not in result


def test_continuation_budget_exhaustion_keeps_completed_local_parent():
    body, training, feedback = fixture()
    first = quote_transfer_partition(body['staged_research']['baseline'], training)
    budget = SearchBudget(first['required_compute'] + len(feedback) + 1)
    entry = replay_research_program(body, training, budget, feedback_rows=feedback)
    result = entry['staged_research_result']
    assert result['status'] == 'continuation_incomplete' and not result['execution_complete']
    assert result['baseline']['execution_complete'] and result['selected_stage'] == 'baseline'
    assert result['budget_incomplete'] and budget.spent <= budget.limit


def test_worse_bound_continuation_keeps_the_actual_first_stage(monkeypatch):
    import hegel_machine.cognition.transfer_partitions as module
    body, training, feedback = fixture()
    native = module.replay_transfer_partition

    def worse(payload, *args, **kwargs):
        entry = native(payload, *args, **kwargs)
        if payload['transfer_partition']['kind'] == 'Pi1':
            entry['candidate'] = replace(entry['candidate'], program=Program('const', value=1000.))
            entry['training_loss'] = 1e6
        return entry

    monkeypatch.setattr(module, 'replay_transfer_partition', worse)
    entry = replay_research_program(body, training, SearchBudget(100000), feedback_rows=feedback)
    result = entry['staged_research_result']
    assert result['execution_complete'] and result['status'] == 'baseline_retained_after_comparison'
    assert result['selected_stage'] == 'baseline'
    assert entry['candidate'].program.to_dict() == result['baseline']['program']
