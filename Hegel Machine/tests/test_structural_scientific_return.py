"""Measured frozen contrasts drive research, before any later refit can run."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import structural_inquiry
from hegel_machine.cognition.finite_native_author import publish_research_options
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from test_native_structural_discrimination import structural_scene


def paid_scene(tmp_path, *, mismatch=False):
    public, agent, lab, question, previous, _ = structural_scene()
    previous['candidates'] = previous['candidates'][:2]
    previous['exported_models'] = previous['exported_models'][:2]
    contract = structural_inquiry.native_structure_planning_contract(public, agent.observations, previous)
    planned = structural_inquiry.execute_native_condition_research(agent, {
        'structure_planning': contract, 'max_compute': contract['proposal_compute']}, lab, question)
    proposal = planned['structure_planning']['proposed_experiments'][0]
    target = proposal['structural_discrimination']
    witness, = target['forecast_witnesses'].values()
    sequence = proposal['query_action']['arguments']['actions'][1]['sequence']
    delta = max(witness['delta_predictions']) + 2. if mismatch else witness['delta_predictions'][0]
    record = {'action': deepcopy(proposal['query_action']['arguments']['actions'][1]),
        'row_indices': list(range(12, 24)), 'rows': [
            {'features': deepcopy(point), 'history': deepcopy(sequence[:i]),
             'target': delta if i == witness['readout_step'] else 0.}
            for i, point in enumerate(sequence)]}
    agent.observations.append(record)
    agent.rows.extend(Observation(row['features'], row['target'], tuple(row['history']))
                      for row in record['rows'])
    agent.workspace = tmp_path
    agent.costs['observation_cost'] += 12
    child = deepcopy(planned['structure_planning']['research_problem'])
    agent.questions.append(child)
    returned = []
    agent._return_structural_answer = lambda q, answer: returned.append((q['id'], deepcopy(answer)))
    callback = {**deepcopy(proposal['research_continuation']),
        'matched_pairs': [{'reference_record_index': 0, 'record_index': 1}],
        'max_compute': 500_000, 'max_candidates': 3}
    return public, agent, lab, question, child, callback, returned


def test_one_refuted_predictor_returns_before_refitting_and_reuses_the_paid_answer(tmp_path, monkeypatch):
    _, agent, lab, question, child, callback, returned = paid_scene(tmp_path)
    before = agent.costs['compute_cost']
    monkeypatch.setattr(structural_inquiry, 'research_native_conditions',
        lambda *a, **k: pytest.fail('a distinguished frozen contrast forced another fit'))
    actual = structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    result = question['native_condition_research'][-1]
    assert actual['status'] == 'frozen_predictions_compared'
    assert agent.costs['compute_cost'] - before == result['compute_cost'] == 4
    verdict = result['structural_discrimination']['forecast_verdict']
    assert verdict['status'] == 'predictions_distinguished'
    assert verdict['refuted_candidate_refs'] and not verdict['structure_families_refuted']
    assert result['research_decision']['kind'] == 'compare_surviving_fixed_candidates'
    assert returned[0][0] == child['id'] and returned[0][1]['status'] == 'answered'
    assert result['candidates'] == [] and agent.candidate_paths == []
    assert len(question['native_structure_planning'][0]['readout_results']) == 1
    costs = deepcopy(agent.costs)
    reused = structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    assert reused['cache_hit'] and agent.costs == costs


def test_both_refuted_forecasts_are_answered_then_cumulative_scope_enters_real_fit(tmp_path, monkeypatch):
    _, agent, lab, question, child, callback, returned = paid_scene(tmp_path, mismatch=True)
    original = structural_inquiry.research_native_conditions
    seen = []

    def fit(*args, **kwargs):
        assert returned[0][0] == child['id'] and returned[0][1]['status'] == 'answered'
        assert question['native_structure_planning'][0]['readout_results'][0]['verdict']['complete']
        seen.append(deepcopy(kwargs['accumulate_matched_scope']))
        return original(*args, **kwargs)
    monkeypatch.setattr(structural_inquiry, 'research_native_conditions', fit)
    before = agent.costs['compute_cost']
    structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    result = question['native_condition_research'][-1]
    assert seen == [{'control': 'u', 'background': {'u': -.4}, 'record_indices': [0, 1]}]
    assert result['accumulated_matched_scope']['matched_pairs']
    assert result['structural_discrimination']['forecast_verdict']['status'] == 'both_predictions_refuted'
    assert result['search_stages'][0]['compute_cost'] == 4
    assert agent.costs['compute_cost'] - before == result['compute_cost']
    assert not result['structural_discrimination']['refit_return']['structure_families_refuted']


def test_completed_readout_survives_later_fit_failure_without_double_charge(tmp_path, monkeypatch):
    _, agent, lab, question, _, callback, _ = paid_scene(tmp_path, mismatch=True)
    original = structural_inquiry.research_native_conditions
    before = agent.costs['compute_cost']
    def interrupted(*args, **kwargs):
        raise RuntimeError('controlled fit interruption after the paid verdict')
    monkeypatch.setattr(structural_inquiry, 'research_native_conditions', interrupted)
    with pytest.raises(RuntimeError, match='controlled fit interruption'):
        structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    assert agent.costs['compute_cost'] - before == 4
    assert len(question['native_structure_planning'][0]['readout_results']) == 1
    monkeypatch.setattr(structural_inquiry, 'research_native_conditions', original)
    before = agent.costs['compute_cost']
    structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    result = question['native_condition_research'][-1]
    assert result['structural_discrimination']['forecast_verdict']['cache_hit']
    assert result['search_stages'][0]['compute_cost'] == 0
    assert agent.costs['compute_cost'] - before == result['compute_cost']


def test_changed_fitting_basis_reuses_only_the_verdict_not_the_old_research(tmp_path, monkeypatch):
    _, agent, lab, question, _, callback, _ = paid_scene(tmp_path, mismatch=True)
    structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    original = structural_inquiry.research_native_conditions
    calls = []
    def fit(*args, **kwargs):
        calls.append(deepcopy(kwargs['basis']))
        return original(*args, **kwargs)
    monkeypatch.setattr(structural_inquiry, 'research_native_conditions', fit)
    callback['basis'] = [Program('feature', name='u').to_dict()]
    actual = structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    assert not actual.get('cache_hit') and calls == [callback['basis']]
    assert question['native_condition_research'][-1]['search_stages'][0]['compute_cost'] == 0
    assert len(question['native_structure_planning'][0]['readout_results']) == 1


def test_unfunded_frozen_check_does_not_fit_or_claim_answer(tmp_path, monkeypatch):
    _, agent, lab, question, child, callback, returned = paid_scene(tmp_path)
    callback['max_compute'] = 3
    monkeypatch.setattr(structural_inquiry, 'research_native_conditions',
        lambda *a, **k: pytest.fail('fit ran without paying the original frozen check'))
    before = deepcopy(agent.costs)
    result = structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    assert result['status'] == 'search_incomplete'
    assert agent.costs['compute_cost'] == before['compute_cost']
    assert returned[0][0] == child['id'] and returned[0][1]['status'] == 'inconclusive'
    assert not question['native_condition_research'][-1]['structural_discrimination']['forecast_verdict']['complete']


def test_changed_source_meaning_cannot_reuse_a_saved_forecast(tmp_path):
    _, agent, lab, question, _, callback, _ = paid_scene(tmp_path)
    callback['structural_discrimination']['source_question_ref']['meaning_version'] = 2
    before = deepcopy(agent.costs)
    with pytest.raises(ValueError, match='saved forecast'):
        structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    assert agent.costs == before


def test_two_segment_quote_prices_cumulative_preparation_without_running_predictions(monkeypatch):
    public, agent, _, question, previous, _ = structural_scene()
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('quote ran a predictor'))

    def quote():
        option = next(item for item in publish_research_options(agent, public)
            if item.get('work_reason') == 'unplanned_structural_ambiguity')
        unit = option['followup_complete_quote']['research_unit']
        price = unit['structural_preparation']
        assert price['frozen_readout_compute'] == 8
        assert unit['search_preparation_compute'] >= price['compute_cost']
        assert not unit['estimate_is_completion_guarantee']
        return price
    original = quote()
    sequence = [{'u': -.4 if i < 3 else .5} for i in range(12)]
    agent.observations.append({'action': {'kind': 'trajectory', 'channels': [], 'sequence': sequence},
        'row_indices': list(range(12, 24)), 'rows': [
            {'features': point, 'history': deepcopy(sequence[:i]), 'target': 0.}
            for i, point in enumerate(sequence)]})
    agent.rows.extend(Observation(row['features'], row['target'], tuple(row['history']))
                      for row in agent.observations[-1]['rows'])
    previous.update(observed_record_count=2, fit_record_indices=[0, 1], fit_evidence=list(range(24)))
    accumulated = quote()
    assert original['related_paid_pairs'] == 0 and accumulated['related_paid_pairs'] == 1
    assert accumulated['matched_assessment_compute_upper'] > original['matched_assessment_compute_upper']
