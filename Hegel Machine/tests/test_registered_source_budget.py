from copy import deepcopy

import pytest

from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.registered_source_budget import (
    retained_grammar_preparation, four_round_unit_plan, symmetric_complete_budget,
)


def test_interrupted_comparison_keeps_consumed_binding_and_remaining_fair_order():
    from hegel_machine.cognition.registered_source_budget import retained_four_round_progress

    plan = [{'candidate_id': owner, 'search_position': {'layer': 0, 'offset': 1},
             'compute': 10} for owner in ('old', 'new')]
    actions = [{'action': 'frozen_recall_on_registered_parent_duty', 'unit_index': 0,
        'registered_unit': plan[0], 'feedback': {'complete_research_visits': [{
            'candidate_id': 'old', 'search_position': {'layer': 0, 'offset': 1},
            'status': 'complete_visit_executed'}]}},
        {'action': 'execution_failure', 'reason': 'Python time exhausted'}]
    result = retained_four_round_progress(plan, actions)
    assert result['consumed_units'] == 1 and result['compared_units'] == 0
    assert result['comparison_of_committed_work_required']
    assert result['completed_unit_counts_by_group'] == {'old': 1}
    assert result['remaining_units'] == [{**plan[1], 'original_unit_index': 1}]
    actions.insert(1, {'action': 'research_choice', 'truncated': False})
    result = retained_four_round_progress(plan, actions)
    assert result['compared_units'] == 1
    assert not result['comparison_of_committed_work_required']


def test_interrupted_work_cannot_import_another_parent_position():
    from hegel_machine.cognition.registered_source_budget import retained_four_round_progress

    plan = [{'candidate_id': 'old', 'search_position': {'layer': 0, 'offset': 1}}]
    actions = [{'action': 'frozen_recall_on_registered_parent_duty', 'unit_index': 0,
        'registered_unit': plan[0], 'feedback': {'complete_research_visits': [{
            'candidate_id': 'old', 'search_position': {'layer': 3, 'offset': 8},
            'status': 'complete_visit_executed'}]}}]
    with pytest.raises(ValueError, match='independent parent cursor'):
        retained_four_round_progress(plan, actions)
from test_complete_revision_visit import _scene
from test_revision_fit_reuse import _retained


def test_frontier_quote_uses_existing_preparation_without_prediction_or_solve(monkeypatch):
    candidate, parent, kwargs, _, _ = _scene()
    prepared = revisions.prepare_revision_visit(candidate, parent, [], {'compute': 5000}, **kwargs)
    cursor = deepcopy(prepared['cursor'])
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('quote repeated prediction'))
    monkeypatch.setattr(revisions, '_fit_revision_binding', lambda *a, **k: pytest.fail('quote called fitter'))
    monkeypatch.setattr(revisions, '_revision_fitting_pools', lambda *a, **k: pytest.fail('quote rebuilt grammar'))
    budget = SearchBudget(10000)
    plan = revisions.quote_revision_frontier(prepared, 4, budget)
    assert len(plan) == 4 and plan[0]['quote'] == prepared['quote']
    assert plan[0]['position'] == {'layer': 0, 'offset': 1}
    assert prepared['cursor'] == cursor and 'pending_fits' not in cursor
    assert all(row['future_fit_query_compute_bound'] > 0 for row in plan if row['quote'])
    prepared['prepared_visit']['_generator'].close()


def test_future_fit_index_and_reuse_work_is_covered_by_growth_bound():
    candidate, parent, kwargs, _, _ = _scene()
    prepared = revisions.prepare_revision_visit(candidate, parent, [], {'compute': 5000}, **kwargs)
    context = prepared['_quote_context']
    template = context['first_template']
    record = _retained(template, context['rows'], scope=parent)
    plan = revisions.quote_revision_frontier(prepared, 1, SearchBudget(10000))
    budget = SearchBudget(10000)
    index = revisions._retained_fit_index([record], budget)
    assert revisions._reuse_revision_fit(template, context['rows'], .5, parent, [record], budget, index=index)
    assert budget.spent < plan[0]['future_fit_query_compute_bound']
    prepared['prepared_visit']['_generator'].close()


def test_pure_grammar_reuse_keeps_fresh_progress_and_shared_source_reference():
    candidate, parent, kwargs, _, _ = _scene()
    first = revisions.prepare_revision_visit(candidate, parent, [], {'compute': 5000}, **kwargs)
    state = deepcopy(first['prepared_visit']['_generator'].gi_frame.f_locals['next_cursor']['fitting'])
    first['prepared_visit']['_generator'].close()
    descriptor = {'source_ref': {'file': 'old.json', 'json_path': ['cursor', 'fitting']},
        'state': {key: state[key] for key in ('structures', 'grammar_cursor')}}
    prepared = revisions.prepare_revision_visit(candidate, parent, [], {'compute': 5000},
        grammar_preparation=descriptor, **kwargs)
    assert prepared['search_position'] == {'layer': 0, 'offset': 1}
    assert 'pending_fits' not in prepared['cursor']
    fitting = prepared['prepared_visit']['_generator'].gi_frame.f_locals['next_cursor']['fitting']
    assert fitting['grammar_preparation_ref'] == descriptor['source_ref']
    assert 'structures' not in fitting and 'grammar_cursor' not in fitting
    assert prepared['quote'] == first['quote']
    prepared['prepared_visit']['_generator'].close()


@pytest.mark.parametrize('changed', ['rows', 'criterion', 'components', 'fitting', 'partial'])
def test_grammar_reuse_refuses_changed_scientific_inputs(changed):
    request = {'components': [{'id': 'seed', 'version': 1}],
        'evidence_checks': [{'source_ref': {'file': 'paid.json', 'json_path': [0]},
            'context_ref': {'reset': 'per_trajectory'}, 'criterion': {'mse_threshold': 1e-10}}],
        'fitting': {'max_structures': 1, 'operations': ['refit']}}
    components = deepcopy(request['components'])
    checks = deepcopy(request['evidence_checks'])
    fitting = deepcopy(request['fitting'])
    state = {'structures': [{}], 'grammar_cursor': {'status': 'searching'}}
    if changed == 'rows': checks[0]['source_ref']['json_path'] = [1]
    if changed == 'criterion': checks[0]['criterion']['mse_threshold'] = 1e-3
    if changed == 'components': components[0]['version'] = 2
    if changed == 'fitting': fitting['operations'] = ['relation']
    if changed == 'partial': state['structures'] = []
    with pytest.raises(ValueError):
        retained_grammar_preparation(request, state, components=components,
            evidence_checks=checks, fitting=fitting, source_ref={'file': 'old.json'})


def test_fair_plan_preserves_group_order_skips_aliases_and_dynamic_query_cost():
    quote = {'compute': 12, 'maximum_outputs': 3}
    groups = [['absent'], ['old', 'alias'], ['new']]
    information = {
        key: {'outer_preparation_compute': 2, 'preparation_compute': 3,
              'paid_rows': 4, 'fit_contract': {'body': 'same'},
              'frontier': [{'position': {'layer': i, 'offset': 1},
                  'status': 'quoted', 'quote': quote, 'quotation_compute': 1,
                  'future_fit_query_compute_bound': 7} for i in range(4)]}
        for key in ('old', 'new')}
    information['absent'] = {'outer_preparation_compute': 2, 'preparation_compute': 3}
    plan = four_round_unit_plan(groups, information)
    assert [item['candidate_id'] for item in plan] == ['old', 'new'] * 4
    assert [item['future_fit_query_upper_compute'] for item in plan] == list(range(0, 56, 7))
    assert plan[0]['source_skips'] == [{'candidate_id': 'absent', 'compute': 5}]
    assert plan[2]['source_skips'] == [{'candidate_id': 'absent', 'compute': 5}]
    assert plan[0]['candidate_ids'] == ['old', 'alias']
    totals = symmetric_complete_budget({'remove': plan[:-1], 'keep': plan}, unpaid_scoring_compute=4)
    assert totals['additional_compute_per_arm'] == sum(x['compute'] for x in plan) + 6


def test_conditional_profiles_reserve_every_possible_future_fit_batch():
    group = [['conditional']]
    info = {'outer_preparation_compute': 1, 'preparation_compute': 2,
        'paid_rows': 97, 'fit_contract': {}, 'frontier': [
            {'position': {'layer': 0, 'offset': i}, 'status': 'quoted',
             'quote': {'compute': 100, 'maximum_outputs': 105},
             'future_fit_query_compute_bound': 10} for i in (1, 2)]}
    plan = four_round_unit_plan(group, {'conditional': info}, rounds=2)
    assert plan[1]['future_fit_query_upper_compute'] == 35 * 10
