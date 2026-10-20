"""Paired task and archive wiring, without scientific solutions or model calls."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_method_learning import register_published_method
from scripts import run_anonymous_revision_choice as study
from test_recurrent_method_adoption import STOP_METHOD, source_study


def test_protocol_has_matching_short_evidence_and_distinguishable_long_paths(tmp_path):
    source_study(tmp_path, 'recurrent')
    plan = study.protocol(tmp_path)
    assert len(plan['cases']) == 4 and plan['arms'] == ['code_react', 'hegel']
    assert plan['limits']['execution_seconds'] == 3600.
    assert plan['limits']['max_model_calls'] == 96
    assert plan['limits']['token_budget'] == 10_000_000
    assert plan['limits']['observation_budget'] == 1152
    assert plan['limits']['confirmation_rows'] == 96
    assert plan['model'] == 'gpt-6-luna'
    assert plan['provider']['timeout_seconds'] == 600.
    assert plan['provider']['reasoning_effort'] == 'medium'
    assert {case['apparatus_seed'] for case in plan['cases']} == {71231031, 71232040}
    for block in range(2):
        pair = [case for case in plan['cases'] if case['parameter_pair'] == block]
        assert {case['kind'] for case in pair} == {'finite2', 'recurrent'}
        assert len({case['prefix_seed'] for case in pair}) == 1
        labs = [study.representation_laboratory(case) for case in pair]
        assert labs[0].public_spec() == labs[1].public_spec()
        rng = Random(pair[0]['prefix_seed'])
        actions = [{'kind': 'trajectory', 'channels': [],
                    'sequence': [{'u': rng.uniform(-1., 1.)} for _ in range(3)]}
                   for _ in range(24)]
        values = [[row['target'] for action in actions for row in lab.execute(action)['rows']]
                  for lab in labs]
        assert max(abs(a-b) for a, b in zip(*values)) < 1e-12
        # Host-only task check: this control sequence is never supplied to an actor.
        probe = {'kind': 'trajectory', 'channels': [],
                 'sequence': [{'u': .8}] + [{'u': 0.} for _ in range(23)]}
        values = [[row['target'] for row in lab.execute(probe)['rows']] for lab in labs]
        assert max(abs(a-b) for a, b in zip(*values)) > .01


def test_all_eight_consumers_receive_only_source_program_and_paid_prefix(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'choice'
    source_study(source, 'recurrent')
    original_source = (source / 'author/training.json').read_text()
    requests = []

    class StopProvider:
        def __init__(self):
            self.calls = 0

        def next_action(self, request):
            self.calls += 1
            if self.calls == 1:
                requests.append(deepcopy(request))
                return {'action': 'run', 'arguments': {'operator': 'procedure',
                    'path': METHOD_PATH, 'max_steps': 12, 'confirm': True},
                    'usage': {'backend': 'scripted_test', 'total_tokens': 1}}
            return {'action': 'provider_error', 'error': {'fixture': 'End scripted test'},
                    'usage': {'backend': 'scripted_test', 'total_tokens': 1}}

    summary = study.run_choice(source, output, provider_factory=lambda workspace: StopProvider(),
                               auth_mode='scripted_fixture')
    assert len(summary['trials']) == len(requests) == 8
    assert summary['evidence_kind'] == 'scripted_interface_check'
    for request in requests:
        assert request['files'][METHOD_PATH] == STOP_METHOD
        assert request['costs']['observation_cost'] == 72
        assert request['costs']['measurement_cost'] == 0
        assert len(request['laboratory_observations']) == 24
        assert all(len(record['rows']) == 3 for record in request['laboratory_observations'])
        assert 'private_terminal_marker' not in json.dumps(request)
        assert 'private_target_fit' not in json.dumps(request)
        assert 'finite2' not in json.dumps(request)
    for row in summary['trials']:
        assert row['published_method_entry_attempts'] == 1
        assert row['published_method_executed_steps'] == row['published_method_stops'] == 1
        assert not row['published_method_confirmed']
        assert row['published_method_present_at_end']
        assert row['published_method_changed_at_end'] is False
        assert row['experiment_comparison_attempts'] == row['prospective_revision_attempts'] == 0
        assert row['acquired_prospective_revisions'] == 0
        assert not row['terminal']['passed']
        assert row['costs']['model_calls'] == 2
        assert row['costs']['observation_cost'] == 72
    plan = json.loads((output / 'protocol.json').read_text())
    assert plan['provider']['auth_mode'] == 'scripted_fixture'
    assert plan['execution_workers'] == 2
    snapshot = output / 'source_at_run/scripts/run_anonymous_revision_choice.py'
    assert snapshot.read_text() == (study.ROOT / 'scripts/run_anonymous_revision_choice.py').read_text()
    assert (source / 'author/training.json').read_text() == original_source


def frozen_source(path):
    source_study(path, 'recurrent')
    plan = json.loads((path / 'protocol.json').read_text())
    plan['method_steps'] = 32
    (path / 'protocol.json').write_text(json.dumps(plan))
    author = json.loads((path / 'author/training.json').read_text())
    memory = register_published_method(
        json.loads((path / 'initial_archive.json').read_text()), author, max_steps=32)
    (path / 'published_result.json').write_text(json.dumps(author))
    (path / 'transfer_archive.json').write_text(json.dumps(memory))
    # The live entry must use the small frozen files, not the paid author export.
    (path / 'author/training.json').write_text('paid author history must not be loaded')
    return memory


def structural_case():
    return {'id': 'source_000', 'kind': 'nested_response', 'seed': 188901031,
            'prefix_seed': 189001034, 'confirmation_seed': 189101034,
            'evaluation_seed': 189201038}


def test_single_frozen_source_uses_small_publication_and_public_inputs(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    memory = frozen_source(source)
    case = structural_case()
    requests, labs = [], []

    def laboratory(selected):
        assert selected == case
        lab = AnonymousHistoryLab(kind='memoryless', seed=case['seed'])
        labs.append(lab)
        return lab

    class InspectProvider:
        def next_action(self, request):
            requests.append(deepcopy(request))
            return {'action': 'provider_error', 'error': {'fixture': 'Stop inspection'},
                    'usage': {'backend': 'scripted_test', 'total_tokens': 1}}

    monkeypatch.setattr(study, 'laboratory', laboratory)
    output = tmp_path / 'online'
    summary = study.run_choice(source, output, cases=[case],
        provider_factory=lambda _: InspectProvider(), auth_mode='scripted_fixture')
    plan = json.loads((output / 'protocol.json').read_text())
    assert plan['arms'] == ['hegel'] and plan['execution_workers'] == 1
    assert plan['method_steps'] == 32
    assert plan['provider']['reasoning_effort'] == 'high'
    assert plan['provider']['timeout_seconds'] == 900.
    assert plan['limits']['max_model_calls'] == 48
    assert plan['limits']['token_budget'] == 12_000_000
    assert plan['limits']['execution_seconds'] == 7200.
    assert plan['limits']['compute_budget'] == 2_000_000
    assert plan['limits']['observation_budget'] == 4096
    assert len(summary['trials']) == len(requests) == len(labs) == 1
    request, = requests
    assert request['session_knowledge'] == memory
    assert request['files'][METHOD_PATH] == STOP_METHOD
    entry, = request['session_knowledge']['manifest']['research_programs']
    assert entry['max_steps'] == 32
    assert request['costs']['observation_cost'] == 72
    public = json.dumps(request)
    for private in ('nested_response', 'apparatus_seed', 'evaluation_seed', str(case['seed']),
                    'private_terminal_marker', 'private_target_fit'):
        assert private not in public
    workspace = output / case['id'] / 'hegel'
    assert (workspace / 'active.json').is_symlink()
    assert (workspace / 'active.json').read_text() == (workspace / 'training.json').read_text()
    assert (source / 'author/training.json').read_text() == 'paid author history must not be loaded'


def test_failed_source_restores_candidates_paid_costs_and_confirmation_stream(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    memory = frozen_source(source)
    case = structural_case()
    prior_workspace = tmp_path / 'failed_source'
    prior = CommonArchiveInquiryAgent(None, prior_workspace, arm='hegel', knowledge=memory,
        source_task=False, confirmation_seed=case['confirmation_seed'],
        config=InquiryLLMConfig(observation_budget=576, confirmation_rows=96,
            confirmation_mse_threshold=1e-6, execution_seconds=30., compute_budget=20000))
    lab = AnonymousHistoryLab(kind='memoryless', seed=case['seed'])
    actions = [{'kind': 'trajectory', 'sequence': [{'u': 0.}] * 24, 'channels': []}] * 3
    feedback = prior._execute('query', {'actions': actions}, lab)
    value = feedback['observations'][0]['rows'][0]['target']
    for action, arguments in [
        ('write', {'path': 'candidate.py', 'code': f'def predict(row): return {value!r}\n'}),
        ('evaluate', {'model': 'candidate.py'}),
        ('finish', {'model': 'candidate.py'}),
    ]:
        result = prior._execute(action, arguments, lab)
        prior.events.append({'step': -1, 'action': action, 'arguments': arguments,
            'feedback': result, 'usage': {'backend': 'zero_llm_fixture', 'total_tokens': 0}})
    assert result['fresh_confirmation']['passed'] is False
    prior.research_candidates['candidate.py'] = {'status': 'paid_refutation'}
    prior._prepare_program_state()
    previous = prior.result('confirmation_failed')
    previous_path = prior_workspace / 'training.json'
    previous_path.write_text(json.dumps(previous))
    original = previous_path.read_text()
    previous = json.loads(original)
    requests = []

    class InspectProvider:
        def next_action(self, request):
            requests.append(deepcopy(request))
            return {'action': 'provider_error', 'error': {'fixture': 'Stop after restore'},
                    'usage': {'backend': 'scripted_test', 'total_tokens': 1}}

    def laboratory(selected):
        assert selected == case
        restored = AnonymousHistoryLab(kind='memoryless', seed=case['seed'])
        # There is no acquisition in this provider: restore must not repurchase.
        restored.execute = lambda _: (_ for _ in ()).throw(AssertionError('unexpected repurchase'))
        return restored

    original_restore = CommonArchiveInquiryAgent.restore_practice_research

    def restore(agent, previous_result, workspace, apparatus, **kwargs):
        outcome = original_restore(agent, previous_result, workspace, apparatus, **kwargs)
        expected = Random(case['confirmation_seed'])
        apparatus.confirmation_actions(96, expected)
        assert apparatus.confirmation_actions(96, deepcopy(agent.confirmation_rng)) == (
            apparatus.confirmation_actions(96, expected))
        assert agent.research_candidates == previous['research_candidates']
        assert agent.observations == previous['paid_observations']
        assert agent.costs == previous['costs']
        assert (agent.workspace / 'candidate.py').read_text() == (prior_workspace / 'candidate.py').read_text()
        return outcome

    monkeypatch.setattr(study, 'laboratory', laboratory)
    monkeypatch.setattr(CommonArchiveInquiryAgent, 'restore_practice_research', restore)
    output = tmp_path / 'online'
    summary = study.run_choice(source, output, cases=[case], resume_result=previous_path,
        provider_factory=lambda _: InspectProvider())
    row, = summary['trials']
    plan = json.loads((output / 'protocol.json').read_text())
    assert row['prior_research_cost'] == previous['costs']
    assert row['costs']['observation_cost'] == 168
    assert row['continuation_cost']['observation_cost'] == 0
    assert row['continuation_cost']['compute_cost'] == 0
    assert row['continuation_cost']['model_calls'] == 1
    assert row['continuation_cost']['reported_tokens'] == 1
    assert plan['limits']['observation_budget'] == 4096 + 168
    assert plan['limits']['compute_budget'] == 2_000_000 + previous['costs']['compute_cost']
    assert plan['online_allowance']['observation_budget'] == 4096
    assert requests[0]['laboratory_observations'] == previous['paid_observations']
    assert previous_path.read_text() == original


@pytest.mark.parametrize('schedule_inquiry', [False, True])
def test_online_continuation_retains_its_actual_context_without_rebuying_prefix(tmp_path, monkeypatch,
                                                                             schedule_inquiry):
    source = tmp_path / 'source'
    memory = frozen_source(source)
    case = structural_case()
    workspace = tmp_path / 'paid_online'
    prior = CommonArchiveInquiryAgent(None, workspace, arm='hegel', knowledge=memory,
        source_task=False, confirmation_seed=case['confirmation_seed'],
        config=InquiryLLMConfig(observation_budget=576, confirmation_rows=96,
            execution_seconds=30., compute_budget=20000))
    lab = AnonymousHistoryLab(kind='memoryless', seed=case['seed'])
    actions = [{'kind': 'trajectory', 'sequence': [{'u': .2}] * 8, 'channels': []}]
    feedback = prior._execute('query', {'actions': actions}, lab)
    prior.events.append({'step': 0, 'action': 'query', 'arguments': {'actions': actions},
        'feedback': feedback, 'usage': {'backend': 'scripted_test', 'total_tokens': 1}})
    prior.costs.update(model_calls=1, reported_tokens=1)
    prior.knowledge['lessons'] = ['retained online interpretation']
    prior._save('research_notes.md', 'Current paid investigation remains unresolved.')
    (workspace / 'research_context.json').write_text(json.dumps(prior._request(lab)))
    previous = prior.result('model_call_budget_exhausted')
    previous_path = workspace / 'training.json'
    previous_path.write_text(json.dumps(previous))
    previous = json.loads(previous_path.read_text())
    requests = []

    class InspectProvider:
        def next_action(self, request):
            requests.append(deepcopy(request))
            return {'action': 'provider_error', 'error': {'fixture': 'Stop after restore'},
                    'usage': {'backend': 'scripted_test', 'total_tokens': 1}}

    def laboratory(selected):
        assert selected == case
        restored = AnonymousHistoryLab(kind='memoryless', seed=case['seed'])
        restored.execute = lambda _: (_ for _ in ()).throw(AssertionError('unexpected repurchase'))
        return restored

    monkeypatch.setattr(study, 'laboratory', laboratory)
    # An online scene must use its actual saved request, not the frozen-practice path.
    monkeypatch.setattr(CommonArchiveInquiryAgent, 'restore_practice_research',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('lost online context')))
    output = tmp_path / 'continued'
    inquiry = ({'question': {'id': 'local_revision', 'question': 'Revise from your paid evidence',
                            'parent_id': 'task', 'depends_on': []},
                'max_model_calls': 4} if schedule_inquiry else None)
    summary = study.run_choice(source, output, cases=[case], resume_result=previous_path,
        provider_factory=lambda _: InspectProvider(), initial_inquiry=inquiry)
    assert requests[0]['session_knowledge']['lessons'] == ['retained online interpretation']
    if schedule_inquiry:
        event = requests[0]['events'][len(previous['events'])]
        assert event['origin'] == 'host_scheduled_development'
        assert event['feedback']['status'] == 'inquiry_started'
        assert event['arguments']['question'] == inquiry['question']
    else:
        assert requests[0]['research_notes'] == 'Current paid investigation remains unresolved.'
    assert requests[0]['laboratory_observations'] == previous['paid_observations']
    assert requests[0]['events'][:len(previous['events'])] == previous['events']
    assert summary['trials'][0]['continuation_cost']['observation_cost'] == 0
    assert summary['trials'][0]['continuation_cost']['model_calls'] == 1
    plan = json.loads((output / 'protocol.json').read_text())
    assert plan['resume_mode'] == 'online_research'
    assert plan.get('initial_inquiry') == inquiry
    assert previous_path.read_text() == json.dumps(previous)


def source_cohort():
    return [{'id': f'source_{index:03d}', 'kind': kind, 'seed': 188911031 + index,
             'apparatus_seed': 188911031 + index, 'prefix_seed': 189011034 + index,
             'confirmation_seed': 189111034 + index, 'evaluation_seed': 189211038 + index}
            for index, kind in enumerate(('recurrent', 'finite2', 'nested_response'))]


def test_independent_cohort_preserves_matched_archives_prefixes_and_private_cases(tmp_path, monkeypatch):
    source, output = tmp_path / 'source', tmp_path / 'cohort'
    memory = frozen_source(source)
    cases = source_cohort()
    original_cases = deepcopy(cases)
    initial_requests = {}
    saved_laboratory = study.laboratory

    def laboratory(case):
        if case['kind'] == 'nested_response':
            return AnonymousHistoryLab(kind='memoryless', seed=case['seed'])
        return saved_laboratory(case)

    class InspectProvider:
        def __init__(self, workspace):
            self.key = (workspace.parent.name, workspace.name)
            self.calls = 0

        def next_action(self, request):
            self.calls += 1
            if self.calls == 1:
                initial_requests[self.key] = deepcopy(request)
                return {'action': 'write', 'arguments': {
                    'path': 'shared/own_source_note.md', 'code': repr(self.key)},
                    'usage': {'backend': 'scripted_test', 'total_tokens': 1}}
            return {'action': 'provider_error', 'error': {'fixture': 'Stop after own edit'},
                    'usage': {'backend': 'scripted_test', 'total_tokens': 1}}

    monkeypatch.setattr(study, 'laboratory', laboratory)
    summary = study.run_choice(source, output, cases=cases,
        provider_factory=InspectProvider, auth_mode='scripted_fixture')
    plan = json.loads((output / 'protocol.json').read_text())
    assert cases == original_cases
    assert plan['arms'] == ['code_react', 'hegel']
    assert [case['arms'] for case in plan['cases']] == [
        ['code_react', 'hegel'], ['hegel', 'code_react'], ['code_react', 'hegel']]
    assert plan['execution_workers'] == 4
    assert plan['model'] == 'gpt-6-luna'
    assert plan['provider']['reasoning_effort'] == 'high'
    assert plan['provider']['timeout_seconds'] == 1800.
    assert plan['method_formation']['status'] == 'planned_after_source_research'
    assert {key: plan['limits'][key] for key in (
        'max_model_calls', 'token_budget', 'execution_seconds', 'observation_budget',
        'compute_budget', 'confirmation_rows', 'confirmation_mse_threshold')} == {
            'max_model_calls': 96, 'token_budget': 12_000_000, 'execution_seconds': 7200.,
            'observation_budget': 4096, 'compute_budget': 2_000_000,
            'confirmation_rows': 96, 'confirmation_mse_threshold': 1e-6}
    assert len(initial_requests) == len(summary['trials']) == 6
    assert summary['source_count'] == 3
    assert summary['evidence_kind'] == 'scripted_interface_check'
    for case in cases:
        requests = [initial_requests[case['id'], arm] for arm in plan['arms']]
        assert requests[0]['laboratory_observations'] == requests[1]['laboratory_observations']
        for request in requests:
            assert request['session_knowledge'] == memory
            assert request['research_phase'] == 'continuation'
            assert request['limits'] == plan['limits']
            assert request['costs']['observation_cost'] == 72
            assert request['costs']['measurement_cost'] == 0
            assert request['files'][METHOD_PATH] == STOP_METHOD
            assert 'shared/own_source_note.md' not in request['files']
            public = json.dumps(request)
            for private in ('apparatus_seed', 'evaluation_seed', str(case['seed']),
                            'private_terminal_marker', 'private_target_fit', case['kind']):
                assert private not in public
            records = request['laboratory_observations']
            expected_lengths = [3] * 24 if case['kind'] in {'finite2', 'recurrent'} else [24] * 3
            assert [len(record['rows']) for record in records] == expected_lengths
        for arm in plan['arms']:
            archive = json.loads((output / case['id'] / arm / 'archive.json').read_text())
            assert archive['files']['shared/own_source_note.md'] == repr((case['id'], arm))
    assert json.loads((source / 'transfer_archive.json').read_text()) == memory


@pytest.mark.parametrize('as_list', [False, True])
def test_case_file_accepts_one_object_or_registered_cohort(tmp_path, monkeypatch, as_list):
    source = tmp_path / 'source'
    frozen_source(source)
    value = source_cohort() if as_list else structural_case()
    case_file = tmp_path / 'cases.json'
    case_file.write_text(json.dumps(value))
    called = []

    def run_choice(*args, **kwargs):
        called.append(kwargs['cases'])
        return {'trials': []}

    monkeypatch.setattr(study, 'run_choice', run_choice)
    monkeypatch.setattr(study, 'verify_codex_subscription', lambda: 'scripted_fixture')
    monkeypatch.setattr('sys.argv', ['run_anonymous_revision_choice.py', '--source', str(source),
        '--output', str(tmp_path / 'output'), '--case-file', str(case_file)])
    study.main()
    assert called == [value if as_list else [value]]


def test_cohort_cannot_restore_one_arms_paid_scene(tmp_path):
    source = tmp_path / 'source'
    frozen_source(source)
    with pytest.raises(ValueError, match='restore one explicitly registered source and arm'):
        study.run_choice(source, tmp_path / 'output', cases=source_cohort(),
            resume_result=tmp_path / 'unopened_checkpoint.json', provider_factory=None)


def test_method_dispatch_keeps_source_failures_and_routes_only_own_confirmed_research(tmp_path, monkeypatch):
    from scripts import run_native_method_compilation as compilation

    source, output = tmp_path / 'frozen_source', tmp_path / 'source_cohort'
    output.mkdir()
    trials = [
        {'case': 'source_000', 'arm': arm, 'status': 'completed',
         'fresh_confirmation_passed': True, 'terminal': {'passed': True}}
        for arm in ('code_react', 'hegel')]
    trials += [
        {'case': 'source_001', 'arm': 'code_react', 'status': 'model_call_budget_exhausted',
         'fresh_confirmation_passed': False, 'terminal': {'passed': False}},
        {'case': 'source_001', 'arm': 'hegel', 'status': 'completed',
         'fresh_confirmation_passed': True, 'terminal': {'passed': False}}]
    prepared, compiled = [], []
    provider = object()

    def prepare(online_study, frozen_study, destination, *, case_id, arm):
        assert online_study == output
        assert frozen_study == source
        assert destination == output / 'methods' / case_id / arm
        prepared.append((case_id, arm))
        return {'source_comparison': {'case': case_id, 'arm': arm},
                'author_only': True, 'transfer_cases': []}

    def run_compilation(plan, destination, *, provider_factory):
        assert provider_factory is provider
        assert plan['author_only'] and plan['transfer_cases'] == []
        comparison = plan['source_comparison']
        compiled.append((comparison['case'], comparison['arm']))
        changed = comparison['arm'] == 'hegel'
        return {'status': 'author_development_completed', 'method_published': True,
                'author_costs': {'model_calls': 2 if changed else 1},
                'source_comparison': {**comparison, 'source_changed': changed,
                    'outcome': 'revised_source_published' if changed else 'unchanged_method_retained'}}

    monkeypatch.setattr(compilation, 'prepare_source_comparison_compilation', prepare)
    monkeypatch.setattr(compilation, 'run_compilation', run_compilation)
    monkeypatch.setattr(study, 'run_study',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('unexpected new investigation')))
    result = study.run_source_method_comparison(source, output, {'trials': trials},
                                               provider_factory=provider)
    assert sorted(prepared) == sorted(compiled) == [
        ('source_000', 'code_react'), ('source_000', 'hegel')]
    assert result['registered_source_episodes'] == len(result['trials']) == 4
    assert [(row['case'], row['arm']) for row in result['trials']] == [
        (row['case'], row['arm']) for row in trials]
    retained, revised, exhausted, terminal_failure = result['trials']
    assert retained['source_comparison']['outcome'] == 'unchanged_method_retained'
    assert retained['source_comparison']['source_changed'] is False
    assert revised['source_comparison']['outcome'] == 'revised_source_published'
    assert revised['source_comparison']['source_changed'] is True
    for failure in (exhausted, terminal_failure):
        assert failure['status'] == 'source_not_confirmed'
        assert failure['method_published'] is False
        assert failure['author_calls'] == 0
        assert 'source_comparison' not in failure
    assert not (output / 'methods' / 'source_001').exists()
    assert json.loads((output / 'method_formation_summary.json').read_text()) == result
    assert json.loads((output / 'campaign_progress.json').read_text()) == {
        'stage': 'completed', 'registered_source_episodes': 4}


def test_interrupted_cohort_retains_finished_runs_and_recovers_only_unaccounted_reply(tmp_path, monkeypatch):
    source, old, output = tmp_path / 'source', tmp_path / 'old', tmp_path / 'resumed'
    memory = frozen_source(source)
    cases = [{**structural_case(), 'id': f'source_{index:03d}'} for index in range(3)]
    plan = study.protocol(source, cases=cases)
    plan['runtime_snapshot'] = []
    plan['method_formation']['status'] = 'scheduled_after_all_source_research'
    old.mkdir()
    (old / 'protocol.json').write_text(json.dumps(plan))
    (old / 'initial_archive.json').write_text(json.dumps(memory))
    retained = {}

    def laboratory(case):
        return AnonymousHistoryLab(kind='memoryless', seed=case['seed'])

    def paid_scene(case, arm, status=None):
        workspace = old / case['id'] / arm
        agent = CommonArchiveInquiryAgent(None, workspace, arm=arm, knowledge=memory,
            source_task=False, confirmation_seed=case['confirmation_seed'],
            config=InquiryLLMConfig(**plan['limits']))
        lab = laboratory(case)
        actions = lab.confirmation_actions(72, Random(case['prefix_seed']))
        feedback = agent._execute('query', {'actions': actions}, lab)
        agent.events.append({'step': -1, 'action': 'query', 'arguments': {'actions': actions},
            'feedback': feedback, 'usage': {'backend': 'common_paid_prefix', 'total_tokens': 0}})
        agent.costs.update(model_calls=3, reported_tokens=100)
        agent._save('model.py', 'def predict(row): return 0.\n')
        (workspace / 'research_context.json').write_text(json.dumps(agent._request(lab)))
        result = agent.result(status or 'researching')
        path = workspace / ('training.json' if status else 'active.json')
        path.write_text(json.dumps(result))
        if status:
            (workspace / 'archive.json').write_text(json.dumps(result['outgoing_knowledge']))
        retained[path] = path.read_text()
        return workspace, result

    paid_scene(cases[0], 'code_react', 'completed')
    paid_scene(cases[0], 'hegel', 'token_budget_exhausted')
    paid_scene(cases[1], 'hegel', 'token_budget_exhausted')
    pending, previous = paid_scene(cases[1], 'code_react')
    payload = {'action': 'evaluate', 'arguments_json': json.dumps({
        'model': 'model.py', 'research_question': 'authored_but_invalid_declaration'}),
        'research_question': None, 'source_code': 'raise AssertionError("must not execute")\n',
        'research_notes': 'Paid notebook survives the host error.'}
    (pending / 'code_action_004.json').write_text(json.dumps(payload))
    (pending / 'code_call_004.jsonl').write_text(json.dumps({'type': 'turn.completed',
        'usage': {'input_tokens': 100, 'output_tokens': 25, 'cached_input_tokens': 80}}) + '\n')
    for name in ('code_action_004.json', 'code_call_004.jsonl', 'model.py'):
        retained[pending / name] = (pending / name).read_text()
    requests, dispatches = {}, []

    class StopProvider:
        def __init__(self, workspace):
            self.key = (workspace.parent.name, workspace.name)
            self.calls = 0

        def next_action(self, request):
            assert self.key not in requests
            requests[self.key] = deepcopy(request)
            if self.key == ('source_001', 'code_react'):
                assert self.calls == 4  # Recovered reply is already accounted for.
            self.calls += 1
            return {'action': 'provider_error', 'error': {'fixture': 'Stop inspection'},
                    'usage': {'backend': 'scripted_test', 'total_tokens': 1}}

    monkeypatch.setattr(study, 'laboratory', laboratory)
    monkeypatch.setattr(study, 'run_source_method_comparison',
        lambda *args, **kwargs: dispatches.append(args[2]))
    result = study.run_choice(None, output, resume_cohort=old, provider_factory=StopProvider)
    assert len(result['trials']) == 6
    assert set(requests) == {('source_001', 'code_react'),
                             ('source_002', 'code_react'), ('source_002', 'hegel')}
    assert dispatches == [result]
    resumed_plan = json.loads((output / 'protocol.json').read_text())
    for field in ('cases', 'arms', 'model', 'provider', 'limits', 'evaluation'):
        assert resumed_plan[field] == plan[field]
    assert resumed_plan['limits']['max_model_calls'] == 96
    assert resumed_plan['limits']['token_budget'] == 12_000_000
    request = requests['source_001', 'code_react']
    assert request['costs']['model_calls'] == 4
    assert request['costs']['reported_tokens'] == 225
    assert request['costs']['observation_cost'] == previous['costs']['observation_cost']
    assert request['costs']['compute_cost'] == previous['costs']['compute_cost']
    assert request['files']['model.py'] == retained[pending / 'model.py']
    assert request['research_notes'] == payload['research_notes']
    event = request['events'][-1]
    assert event['arguments']['code'] == payload['source_code']
    assert event['arguments']['research_question'] == 'authored_but_invalid_declaration'
    assert 'research_question must be a JSON object' in event['feedback']['error']
    assert event['usage']['total_tokens'] == 125
    row, = [row for row in result['trials']
            if row['case'] == 'source_001' and row['arm'] == 'code_react']
    assert row['costs']['model_calls'] == 5
    assert row['costs']['reported_tokens'] == 226
    assert row['prior_research_cost']['model_calls'] == 4
    assert row['continuation_cost']['model_calls'] == 1
    assert row['continuation_cost']['reported_tokens'] == 1
    assert row['recovered_paid_reply']['model_call'] == 4
    for key in (('source_002', 'code_react'), ('source_002', 'hegel')):
        assert requests[key]['costs']['observation_cost'] == 72
        assert requests[key]['costs']['model_calls'] == 0
    original, = [row for row in result['trials']
                 if row['case'] == 'source_000' and row['arm'] == 'code_react']
    assert original['archive'] == str(old / 'source_000/code_react/archive.json')
    assert (output / 'source_000/code_react/active.json').resolve() == (
        old / 'source_000/code_react/training.json')
    assert not (old / 'source_000/code_react/evaluation.json').exists()
    assert not (old / 'source_000/code_react/research_trace.json').exists()
    for path, content in retained.items():
        assert path.read_text() == content


def test_resume_cohort_cli_uses_saved_model_and_needs_no_source_argument(tmp_path, monkeypatch):
    old = tmp_path / 'old'
    old.mkdir()
    plan = {'model': 'saved-model', 'provider': {'reasoning_effort': 'high', 'timeout_seconds': 1800}}
    (old / 'protocol.json').write_text(json.dumps(plan))
    calls = []

    def run_choice(source, output, **kwargs):
        calls.append((source, kwargs['resume_cohort']))
        return {'trials': []}

    monkeypatch.setattr(study, 'run_choice', run_choice)
    monkeypatch.setattr(study, 'verify_codex_subscription', lambda: 'scripted_fixture')
    monkeypatch.setattr('sys.argv', ['run_anonymous_revision_choice.py',
        '--resume-cohort', str(old), '--output', str(tmp_path / 'output')])
    study.main()
    assert calls == [(None, old)]
