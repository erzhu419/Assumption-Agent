"""Protocol and archive wiring; scripted fixtures are not scientific evidence."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.anonymous_method_learning import (
    AnonymousPreparedMethodAgent, SingleSubmissionMethodAgent)
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import run_anonymous_method_revision as revision
from scripts import run_anonymous_method_learning as learning
from test_anonymous_method_learning import action, memoryless_source, read_json, write_json
from test_recurrent_method_revision import SOURCE, revision_sources


@pytest.fixture
def design_source(revision_sources):
    source, repairs = revision_sources
    plan = revision.protocol(source, repair_study=repairs)
    write_json(source / 'protocol.json', plan)
    write_json(source / 'summary.json', {'prior_author_phases': []})
    return source


def test_design_protocol_requalifies_practice_and_keeps_only_original_six_targets(design_source):
    old = read_json(design_source / 'protocol.json')
    plan = revision.protocol(design_source, design_experiments=True)
    assert list(plan['practice_cases']) == [f'probe_case_{i:03d}' for i in range(6)]
    assert list(plan['practice_cases'].values()) == list(old['practice_cases'].values())
    assert set(plan['practice_cases']).isdisjoint(old['practice_cases'])
    assert plan['transfer_cases'][:6] == old['transfer_cases'][:6]
    assert plan['old_target_ids'] == old['old_target_ids']
    fresh = plan['transfer_cases'][6:]
    assert len(fresh) == 4
    assert {case['apparatus_seed'] for case in fresh} == {91231031, 91232040}
    assert {(case['parameter_pair'], case['kind']) for case in fresh} == {
        (block, kind) for block in (0, 1) for kind in ('recurrent', 'finite2')}
    assert {case['apparatus_seed'] for case in fresh}.isdisjoint(
        case['apparatus_seed'] for case in [*old['practice_cases'].values(), *old['transfer_cases']])
    assert plan['limits']['observation_budget'] == 1152
    assert plan['limits']['confirmation_rows'] == 96
    assert plan['author_limits']['max_model_calls'] == 96
    assert plan['provider']['timeout_seconds'] == 600.
    assert 'repair_study' not in plan
    with pytest.raises(ValueError, match='one fresh author'):
        revision.protocol(design_source, design_experiments=True, repair_study=design_source)


def test_restored_support_is_experience_not_new_protocol_qualification(
        tmp_path, design_source, monkeypatch):
    previous = read_json(design_source / 'author/training.json')
    knowledge = read_json(design_source / 'initial_archive.json')
    plan = revision.protocol(design_source, design_experiments=True)
    plan['author_limits']['max_model_calls'] = 1
    plan['runtime_snapshot'] = []
    output = tmp_path / 'design'

    def not_used(*args, **kwargs):
        raise AssertionError('Do not repurchase an old counterexample, append repair archives, '
                             'or transfer an unpublished method.')

    monkeypatch.setattr(revision, 'reproduce_counterexample', not_used)
    monkeypatch.setattr(revision, 'attach_repair_archive', not_used)
    monkeypatch.setattr(revision, 'frozen_transfers', not_used)

    class InspectAuthor:
        def next_action(self, request):
            assert request['files'][METHOD_PATH] == SOURCE
            assert request['costs']['observation_cost'] == 0
            assert request['session_knowledge']['source_catalog'] == knowledge['source_catalog']
            assert 'FIRST executable model submission ENDS program research' in request['task_interface']
            assert 'Failed confirmation becomes paid evidence on the next invocation.' not in request['task_interface']
            resume = read_json(output / 'author/resume_context.json')
            coverage = resume['current_source_coverage']
            assert coverage['confirmed_cases'] == []
            assert coverage['missing_cases'] == sorted(plan['practice_cases'])
            assert len(resume['previous_practice']) == len(previous['practice_trials'])
            return action('run', {'code': "print('Inspect inherited evidence without new acquisition.')"})

    summary = revision.run_revision(plan, output, provider_factory=lambda _: InspectAuthor())
    result = read_json(output / 'author/training.json')
    assert result['practice_trials'] == previous['practice_trials']
    assert not summary['method_published']
    assert summary['unavailable_target_episodes'] == 20
    assert summary['author_costs']['observation_cost'] == 0
    assert not (output / 'author/repair_archive').exists()
    assert summary['old_author_costs']['observation_cost'] == previous['costs']['observation_cost']


def test_published_comparison_uses_identical_archive_and_stops_all_research_before_scoring(
        tmp_path, design_source, monkeypatch):
    plan = revision.protocol(design_source, design_experiments=True)
    plan['runtime_snapshot'] = []
    knowledge = read_json(design_source / 'initial_archive.json')
    order = []
    revised = SOURCE.replace('fixture source', 'revised fixture source')

    def publish(author, *, on_checkpoint):
        payload = author.prepare_trial(plan['practice_cases']['probe_case_000'],
            author.workspace / 'fixture_practice', revised, InquiryLLMConfig(**plan['limits']))
        assert isinstance(payload['agent'], SingleSubmissionMethodAgent)
        assert payload['agent'].costs['observation_cost'] == 72
        assert payload['agent'].knowledge['source_catalog'] == knowledge['source_catalog']
        assert not (payload['agent'].workspace / 'repair_archive').exists()
        author.model, author.published_source = METHOD_PATH, revised
        return author.result('completed')

    def targets(part, branch, memory, sources, method_source, *, agent_type=AnonymousPreparedMethodAgent):
        single = agent_type is SingleSubmissionMethodAgent
        order.append(('research', branch.name, single))
        assert [case['id'] for case in part['transfer_cases']] == [
            case['id'] for case in plan['transfer_cases']
            if (case['id'] not in plan['old_target_ids']) == single]
        assert memory['source_catalog'] == knowledge['source_catalog']
        assert all(memory['files'][name] == text for name, text in knowledge['files'].items()
                   if name not in {METHOD_PATH, 'shared/knowledge.json'})
        assert memory['files'][METHOD_PATH] == method_source
        return [(case, {'branch': branch.name, 'costs': {'observation_cost': 17, 'compute_cost': 5},
                        'events': [], 'method_results': []}) for case in part['transfer_cases']]

    def score(part, results):
        assert order[:4] == [('research', 'original', False), ('research', 'original', True),
                             ('research', 'revised', False), ('research', 'revised', True)]
        order.append(('score', results[0][1]['branch']))
        return [{'case': case['id'], 'kind': case['kind'], 'status': 'completed',
                 'costs': result['costs'], 'terminal': {'passed': True}}
                for case, result in results]

    monkeypatch.setattr(revision.AnonymousResearchMethodLearner, 'run', publish)
    monkeypatch.setattr(revision, 'frozen_transfers', targets)
    monkeypatch.setattr(revision, 'score_frozen', score)
    summary = revision.run_revision(plan, tmp_path / 'paired', provider_factory=lambda _: None)
    assert summary['method_published'] and summary['source_changed']
    assert summary['unavailable_target_episodes'] == 0
    assert summary['target_costs']['original'] == summary['target_costs']['revised']
    assert [pair['group'] for pair in summary['pairs']] == ['old_target'] * 6 + ['new_parameter'] * 4
    for records in summary['frozen'].values():
        assert [record['research_protocol'] for record in records] == [
            'original_confirmation_feedback'] * 6 + ['single_submission'] * 4
        assert all('experimental_design' in record for record in records)


def test_frozen_factory_reaches_actual_prepared_agent(tmp_path, design_source, monkeypatch):
    plan = revision.protocol(design_source, design_experiments=True)
    plan['transfer_cases'] = plan['transfer_cases'][6:7]
    knowledge = read_json(design_source / 'initial_archive.json')
    sources = read_json(design_source / 'source_outcomes.json')
    executed = []

    def execute(agent, lab, *, max_steps, on_checkpoint):
        executed.append(type(agent))
        assert max_steps == 12
        assert agent.frozen_method_source == SOURCE
        assert agent.costs['observation_cost'] == 72
        return {'status': 'method_stopped', 'costs': deepcopy(agent.costs)}

    monkeypatch.setattr(SingleSubmissionMethodAgent, 'execute_method', execute)
    results = learning.frozen_transfers(plan, tmp_path / 'frozen_factory', knowledge, sources,
        SOURCE, agent_type=SingleSubmissionMethodAgent)
    assert executed == [SingleSubmissionMethodAgent]
    assert len(results) == 1
