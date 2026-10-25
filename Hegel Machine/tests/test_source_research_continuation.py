"""Paid-only continuation boundaries use restored fixture evidence."""

from copy import deepcopy

import pytest

from scripts import prepare_pr3_source_successor as preparation
from scripts import run_anonymous_revision_inquiry as study
from test_source_comparison_continuation import RestorableLab, saved_study


def completed_comparison(tmp_path, monkeypatch):
    source, _, _ = saved_study(tmp_path)
    output = tmp_path / 'comparison'
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **kwargs: RestorableLab())
    monkeypatch.setattr(study, '_aggregate_registered_sources',
                        lambda *args, **kwargs: {'new_compute_cost': 0, 'status': 'UNKNOWN'})
    preparation.prepare_source_comparisons(source, output)
    preparation.run_prepared_source_comparisons(output)
    return output / 'source'


def test_research_registration_keeps_common_caps_reserve_and_original_input(tmp_path, monkeypatch):
    source = completed_comparison(tmp_path, monkeypatch)
    original = preparation._read(source / 'protocol.json')
    summary = preparation._read(source / 'summary.json')
    output = tmp_path / 'research'
    registration = preparation.prepare_source_research(source, output)
    plan = preparation._value(registration['plan_ref'])
    assert plan['limits'] == {**original['limits'], 'compute_budget': 350030}
    assert plan['author'] == original['author']
    assert plan['prepared_public_questions'] == original['prepared_public_questions']
    assert plan['source_cohort_allocation']['aggregation_compute'] == 4
    assert plan['source_cohort_allocation']['source_compute_limit'] == 700060
    assert set(plan['source_cohort_allocation']['source_stage_compute'].values()) == {350028}
    assert registration['native_compute'] == 100000
    assert preparation._value(registration['initial_archive_ref'])['files']
    assert preparation._value(registration['author_source_ref'])
    for job, old in zip(registration['jobs'], summary['trials']):
        assert job['prior_costs'] == old['costs']
        assert job['compute_cap'] == old['compute_cap'] + 200000
        assert job['mode'] == ('retained_completed' if old['mode'] == 'retained_completed'
                               else 'same_paid_source_research')
    assert not (output / 'source').exists()


def test_real_restore_preserves_ledger_and_stops_before_unpaid_author_query(tmp_path, monkeypatch):
    source = completed_comparison(tmp_path, monkeypatch)
    output = tmp_path / 'research'
    registration = preparation.prepare_source_research(source, output)
    # A fixture author really requests acquisition; no acquisition is executed.
    preparation._write(output / 'registered_author.json', {'source':
        "def select_actions(spec, observations, memory):\n"
        "    return {'action':'query','arguments':{'actions':[{'new_fixture_input':1}]}}\n"})
    restored, hooks = [], []
    original_restore = study.RegisteredOrganizationAgent.restore_research

    class NoNewLabels(RestorableLab):
        def execute(self, action):
            pytest.fail('source continuation acquired new labels')

    def restore(agent, previous, *args, **kwargs):
        value = original_restore(agent, previous, *args, **kwargs)
        agent._open_question({'id': 'organization:fixture', 'question': 'Saved theory owner',
                              'parent_id': None, 'depends_on': [], 'queue': 'theory'})
        agent._active_question_id = 'organization:fixture'
        restored.append((agent, deepcopy(previous)))
        return value

    def before(agent, lab):
        assert agent._active_question_id == agent.knowledge['finite_native_author']['question']['id']
        hooks.append(agent.workspace)

    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **kwargs: NoNewLabels())
    monkeypatch.setattr(study.RegisteredOrganizationAgent, 'restore_research', restore)
    monkeypatch.setattr(study.RegisteredOrganizationAgent, '_before_model_turn', before)
    summary = preparation.run_prepared_source_research(output)
    assert len(hooks) == len(restored) == 2
    assert summary['new_observations'] == summary['new_terminal_evaluations'] == 0
    assert summary['continuation_compute_cost'] == 0
    for agent, previous in restored:
        assert agent.knowledge['finite_native_author']['max_compute'] == 100000
        assert agent.costs['compute_cost'] == previous['costs']['compute_cost']
        assert agent.costs['observation_cost'] == previous['costs']['observation_cost']
        assert agent.events == previous['events']
        assert agent.method_results == previous['method_results']
        result = preparation._read(agent.workspace / 'training.json')
        assert result['status'] == 'same_paid_boundary'
        assert result['same_paid_source_continuation']['boundary']['proposed_action']['action'] == 'query'
        assert result['continuation_cost']['model_calls'] == 0
        assert result['model'] is None and result['fresh_confirmation'] is None
    retained = [j for j in summary['trials'] if j['mode'] == 'retained_completed']
    assert len(retained) == 4
    assert all(j['costs'] == j['prior_costs'] for j in retained)
    assert all(j['continuation_cost']['compute_cost'] == 0 for j in retained)
    assert all(not (output / 'source' / j['case'] / j['arm']).exists() for j in retained)
