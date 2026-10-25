"""The paid comparison continuation uses fixture data, never a scientific lab."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.finite_native_author import export_finite_native_author
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from scripts import prepare_pr3_source_successor as preparation
from scripts import run_anonymous_revision_inquiry as study
from test_finite_native_author import ChoiceLab, declaration
from test_inquiry_revision import action


DRIVER = 'shared/finite_native_author.py'
MODELS = ['old/_evaluated_model/model.py', 'new/_evaluated_model/model.py']


class RestorableLab(ChoiceLab):
    def __init__(self):
        super().__init__()
        self.costs = {'observation_cost': 0}


def saved_study(tmp_path):
    source = tmp_path / 'old'
    source.mkdir()
    library = source / 'initial_library.json'
    preparation._write(library, {'candidates': {}})
    contract = declaration()
    contract['predictor_paths'] = MODELS
    knowledge = {'files': {DRIVER: export_finite_native_author(),
        MODELS[0]: 'def predict(row): return 0.\n',
        MODELS[1]: 'def predict(row): return 1.\n'},
        'finite_native_author': contract,
        'theory_consolidation': {'file': str(library), 'json_path': []}}
    organization = {'phase': 'source', 'theories': [], 'initial_library_path': str(library),
        'consolidation': {'source_theory_ids': [],
            'source_question_refs': [{'question_id': contract['question']['id'], 'meaning_version': 1}],
            'allocation': {'compute': 4, 'max_groups': 8},
            'mission': {'purpose': 'theory_learning', 'mission_id': 'fixture',
                        'theory_learning_allocation': {'compute': 4}}}}
    plan = study.protocol(kinds=['fixture'], cases=2, arms=study.ORGANIZATION_ARMS,
        config=InquiryLLMConfig(observation_budget=20, confirmation_rows=2,
            compute_budget=30, execution_seconds=300., max_model_calls=0, token_budget=0),
        research_organization=organization,
        author={'kind': 'finite_native', 'path': DRIVER, 'max_steps': 48, 'confirm': True})
    plan['prepared_public_questions'] = {case['id']: deepcopy(contract['question'])
                                         for case in plan['cases']}
    trials = []
    for case in plan['cases']:
        for arm in plan['arms']:
            if arm != 'pr3':
                costs = {'observation_cost': 2, 'measurement_cost': 0, 'compute_cost': 7,
                         'python_execution_seconds': 1., 'model_calls': 0, 'reported_tokens': 0}
                trials.append({'case': case['id'], 'arm': arm, 'model': 'retained.py',
                    'status': 'completed', 'costs': costs})
                continue
            local = deepcopy(organization)
            local['consolidation']['allocation']['compute'] = 0
            cap = plan['source_cohort_allocation']['source_stage_compute'][case['id']]
            agent = study.RegisteredOrganizationAgent(None, source / case['id'] / arm,
                arm=arm, research_organization=local, knowledge=deepcopy(knowledge),
                config=InquiryLLMConfig(**{**plan['limits'], 'compute_budget': cap}),
                source_task=False)
            agent._organization_state['finite_author'] = {'contract': deepcopy(plan['author']),
                                                         'completed_steps': 2}
            lab = RestorableLab()
            agent._execute('query', {'actions': [action(1., 2.)],
                'research_question': deepcopy(contract['question'])}, lab)
            # The toy old account leaves one full score; the second really fails.
            agent.costs['compute_cost'] = cap - len(agent.rows)
            outcome = execute_procedure(agent, lab, DRIVER, max_steps=2,
                allow_confirmation=False, record_events=True)
            assert outcome['status'] == 'method_no_progress'
            assert len(agent.method_results) == 2
            assert all(row['feedback']['error'].endswith('scoring budget exhausted')
                       for row in agent.method_results)
            result = agent.result(outcome['status'])
            study.write_json(agent.workspace / 'training.json', result)
            trials.append({'case': case['id'], 'arm': arm, 'model': None,
                'status': result['status'], 'costs': result['costs']})
    preparation._write(source / 'protocol.json', plan)
    preparation._write(source / 'initial_archive.json', knowledge)
    preparation._write(source / 'summary.json', {'trials': trials})
    return source, plan, knowledge


def test_preparation_extends_original_caps_and_retains_completed_references(tmp_path, monkeypatch):
    source, original, _ = saved_study(tmp_path)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda *a, **k: pytest.fail('prepare made a lab'))
    monkeypatch.setattr(study.RegisteredOrganizationAgent, 'restore_research',
                        lambda *a, **k: pytest.fail('prepare restored research'))
    output = tmp_path / 'continued'
    registration = preparation.prepare_source_comparisons(source, output)
    plan = preparation._read(output / 'comparison_plan.json')
    assert plan['limits'] == {**original['limits'], 'compute_budget': 150030}
    assert plan['author_source_ref'] == {'file': str(source / 'initial_archive.json'),
        'json_path': ['files', plan['author']['path']]}
    assert preparation._value(plan['author_source_ref'])
    cohort = plan['source_cohort_allocation']
    assert cohort['aggregation_compute'] == 4
    assert cohort['source_compute_limit'] == 300060
    assert set(cohort['source_stage_compute'].values()) == {150028}
    assert cohort['local_consolidation_compute'] == 0
    for job in registration['jobs']:
        assert job['compute_cap'] == job['original_compute_cap'] + 150000
        if job['mode'] == 'retained_completed':
            assert job['training_ref']['file'].startswith(str(source))
            assert job['evaluation_ref']['file'].startswith(str(source))
    assert not (output / 'source').exists()


def test_real_restore_one_author_step_reuses_paid_scores_and_preserves_old_account(tmp_path, monkeypatch, capsys):
    source, _, _ = saved_study(tmp_path)
    output = tmp_path / 'continued'
    preparation.prepare_source_comparisons(source, output)
    restored, aggregated = [], []
    original_restore = study.RegisteredOrganizationAgent.restore_research

    def restore(agent, previous, *args, **kwargs):
        feedback = original_restore(agent, previous, *args, **kwargs)
        restored.append((agent, deepcopy(previous)))
        return feedback

    class NoNewLabels(RestorableLab):
        def execute(self, action):
            pytest.fail('continuation must not purchase observations')

    def aggregate(plan, destination, trials, organization, knowledge, paid_traces, previous_study):
        assert previous_study == source
        assert len(trials) == 2
        assert all(row['result']['research_organization']['consolidation_basis']
                   for row in trials)
        assert sum(row['result']['costs']['compute_cost'] for row in trials) == 60
        assert plan['source_cohort_allocation']['aggregation_compute'] == 4
        aggregated.append(trials)
        return {'new_compute_cost': 0, 'status': 'UNKNOWN'}

    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **kwargs: NoNewLabels())
    monkeypatch.setattr(study.RegisteredOrganizationAgent, 'restore_research', restore)
    monkeypatch.setattr(study.RegisteredOrganizationAgent, '_before_model_turn',
                        lambda *a: pytest.fail('continuation must not route or generate'))
    monkeypatch.setattr(study, '_aggregate_registered_sources', aggregate)
    summary = preparation.run_prepared_source_comparisons(output)
    assert len(restored) == 2 and len(aggregated) == 1
    assert summary['new_observations'] == summary['new_terminal_evaluations'] == 0
    assert summary['continuation_compute_cost'] == 4
    for agent, previous in restored:
        assert agent.costs['compute_cost'] == previous['costs']['compute_cost'] + 2
        assert agent.costs['observation_cost'] == previous['costs']['observation_cost']
        assert agent.costs['model_calls'] == previous['costs']['model_calls'] == 0
        assert agent.observations == previous['paid_observations']
        assert agent.events[:len(previous['events'])] == previous['events']
        assert len(agent.method_results) == len(previous['method_results']) + 1
        feedback = agent.method_results[-1]['feedback']
        assert feedback['operator'] == 'research_choice'
        assert feedback['reused_score_count'] == feedback['new_scored_count'] == 1
        assert [row['path'] for row in feedback['evaluations']] == MODELS
        assert feedback['evaluations'][1]['retention_decision']['status'] == 'candidate_retained'
        assert feedback['evaluations'][0]['paid_score_reuse']['execution_basis_ref']['file'].startswith(
            str(source))
        assert agent._organization_state['finite_author']['completed_steps'] == 3
        assert agent.config.execution_seconds == 300.
        assert agent.model is None and not agent.confirmations
        assert agent.candidate_paths == []
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(progress) == 6
    assert sum(row['continuation_compute_cost'] for row in progress) == 4
    assert sum(row['status'] == 'retained_completed' for row in progress) == 4


def test_changed_author_query_is_not_executed(tmp_path, monkeypatch):
    source, _, _ = saved_study(tmp_path)
    output = tmp_path / 'continued'
    preparation.prepare_source_comparisons(source, output)
    for case in ('case_000', 'case_001'):
        (source / case / 'pr3' / DRIVER).write_text(
            "def select_actions(spec, observations, memory):\n"
            "    return {'action': 'query', 'arguments': {'actions': []}}\n")
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **kwargs: RestorableLab())
    monkeypatch.setattr('hegel_machine.cognition.temporal_paired_method.execute_procedure',
                        lambda *a, **k: pytest.fail('changed author must not dispatch'))
    monkeypatch.setattr(study, '_aggregate_registered_sources',
                        lambda *a, **k: {'new_compute_cost': 0, 'status': 'UNKNOWN'})
    summary = preparation.run_prepared_source_comparisons(output)
    continued = [row for row in summary['trials'] if row['mode'] == 'one_paid_comparison']
    assert all(row['status'] == 'comparison_not_executed' for row in continued)
    assert all(row['continuation_cost']['compute_cost'] == 0 for row in continued)
    assert summary['new_observations'] == 0
