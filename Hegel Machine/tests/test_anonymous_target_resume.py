"""Resume paid research without reauthoring, repurchasing prefixes, or leaking source fits."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts import run_anonymous_method_learning as learning
from scripts import run_anonymous_method_revision as revision
from scripts.experimental_design_author import ExperimentalDesignMethodLearner
from scripts.experimental_design_control import ExperimentalDesignMethodAgent
from test_anonymous_method_learning import memoryless_source, read_json, write_json
from test_conditional_design_study import prospective_source
from test_prospective_design_study import acquisition_source
from test_recurrent_method_revision import SOURCE, revision_sources


def test_completed_failure_is_reused_without_new_practice(tmp_path, monkeypatch):
    case = {'id': 'already_failed'}
    path = tmp_path / 'frozen' / case['id']
    paid = {'workspace': str(path), 'status': 'confirmation_failed',
            'costs': {'observation_cost': 168}, 'method_results': [1, 2]}
    write_json(path / 'training.json', paid)

    def forbidden(*args, **kwargs):
        raise AssertionError('A recorded failed outcome is complete, not a rerun request.')

    monkeypatch.setattr(learning, 'trial_preparer', lambda *a, **k: forbidden)
    results = learning.frozen_transfers({'limits': {}, 'transfer_cases': [case]},
                                       tmp_path, {}, [], SOURCE)
    assert results == [(case, paid)]
    assert read_json(path / 'training.json') == paid


def test_partial_target_retains_steps_files_and_paid_prefix(tmp_path, prospective_source):
    plan = revision.protocol(prospective_source, conditional_design=True)
    case = plan['transfer_cases'][10]
    plan.update(transfer_cases=[case], method_steps=32)
    knowledge = read_json(prospective_source / 'initial_archive.json')
    method = '''import json
from pathlib import Path
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    if len(steps) < 2:
        letter = 'AB'[len(steps)]
        return {'action': 'run', 'arguments': {'code':
            "from pathlib import Path; p=Path('progress.txt'); p.write_text((p.read_text() if p.exists() else '') + " + repr(letter) + ")"}}
    return {'stop': 'finished recorded actions'}
'''
    output = tmp_path / 'partial'
    path = output / 'frozen' / case['id']
    limits = InquiryLLMConfig(**{**plan['limits'], 'max_model_calls': 0, 'token_budget': 0})
    prepare = learning.trial_preparer(knowledge, [], output / 'reference_archive',
                                     agent_type=ExperimentalDesignMethodAgent)
    initial = prepare(case, path, method, limits)
    def interrupt(result):
        write_json(path / 'active.json', result)
        raise InterruptedError('Stop after the first persisted operation.')

    with pytest.raises(InterruptedError):
        initial['agent'].execute_method(initial['laboratory'], max_steps=32,
                                       on_checkpoint=interrupt)
    before = read_json(path / 'active.json')
    assert before['status'] == 'researching'
    assert len(before['method_results']) == 1
    assert (path / 'progress.txt').read_text() == 'A'
    assert before['costs']['observation_cost'] == 72

    results = learning.frozen_transfers(plan, output, knowledge, [], method,
                                       agent_type=ExperimentalDesignMethodAgent)
    result = results[0][1]
    assert result['status'] == 'method_stopped'
    assert len(result['method_results']) == 3
    assert result['costs']['observation_cost'] == 72
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 0
    assert (path / 'progress.txt').read_text() == 'AB'
    retained = next((output / 'interrupted_targets').iterdir())
    assert (retained / 'progress.txt').read_text() == 'A'
    assert read_json(retained / 'active.json') == before
    assert read_json(path / 'target_continuation.json')['remaining_steps'] == 31
    assert read_json(path / 'training.json')['costs'] == result['costs']


def test_partial_without_checkpoint_is_preserved_and_not_replayed(tmp_path):
    case = {'id': 'early_interruption'}
    path = tmp_path / 'frozen' / case['id']
    path.mkdir(parents=True)
    (path / 'paid.txt').write_text('incomplete action record')
    with pytest.raises(ValueError, match='no completed-action checkpoint'):
        learning.frozen_transfers({'limits': {}, 'transfer_cases': [case]},
                                 tmp_path, {}, [], SOURCE)
    assert (path / 'paid.txt').read_text() == 'incomplete action record'
    assert not (path / 'training.json').exists()


def test_online_experience_is_author_only_and_excludes_terminal_evaluation(tmp_path):
    source = tmp_path / 'online'
    write_json(source / 'training.json', {'status': 'completed', 'model': 'shared/model.py',
        'costs': {'observation_cost': 288}, 'fresh_confirmation': {'passed': True, 'mse': 0.0}})
    write_json(source / 'shared/weights.json', {'fit': 'source task only'})
    (source / 'shared/model.py').write_text('def predict(row): return 0.0')
    write_json(source / 'evaluation.json', {'terminal_marker': 'never author input'})
    workspace = tmp_path / 'author'
    workspace.mkdir()
    knowledge = {'files': {'old.py': 'untouched'}, 'source_catalog': []}
    before = deepcopy(knowledge)
    author = SimpleNamespace(workspace=workspace, files={}, knowledge=knowledge,
                             common_interface='existing publication requirements')
    evidence = revision.attach_online_research_experience(author, source)
    assert evidence['fresh_confirmation']['passed']
    assert (workspace / evidence['model']).read_text() == (source / 'shared/model.py').read_text()
    assert (workspace / 'online_research_experience/shared').is_symlink()
    assert not (workspace / 'online_research_experience/evaluation.json').exists()
    assert 'terminal_marker' not in json.dumps(author.files)
    assert author.knowledge == before
    assert set(author.files) == {'online_research_experience.json'}
    assert 'existing publication requirements' in author.common_interface


def test_published_target_entry_finishes_all_56_before_scoring_without_provider(
        tmp_path, prospective_source, monkeypatch):
    plan = revision.protocol(prospective_source, conditional_design=True)
    plan['runtime_snapshot'] = []
    output = tmp_path / 'published_before_targets'

    def publish(author, **kwargs):
        author.model = 'shared/consolidated_method.py'
        author.published_source = SOURCE.replace('fixture source', 'revised source')
        return author.result('completed')

    actual_targets = revision.run_revision_targets
    monkeypatch.setattr(ExperimentalDesignMethodLearner, 'run', publish)
    monkeypatch.setattr(revision, 'run_revision_targets',
                        lambda *args, **kwargs: {'paused_after_publication': True})
    revision.run_revision(plan, output, provider_factory=lambda _: None)
    assert read_json(output / 'published_method.json')['published']
    monkeypatch.setattr(revision, 'run_revision_targets', actual_targets)
    researched, scored = [], []
    old_ids = set(plan['old_target_ids'])
    new_ids = {case['id'] for case in plan['transfer_cases']} - old_ids

    def targets(part, branch, memory, sources, method_source, **kwargs):
        actual = {case['id'] for case in part['transfer_cases']}
        assert actual == (old_ids if branch.name == 'original' else
                          old_ids | new_ids if branch.name == 'revised' else new_ids)
        researched.append(branch.name)
        return [(case, {'branch': branch.name, 'events': [], 'method_results': [],
                        'costs': {'observation_cost': 72, 'compute_cost': 1}})
                for case in part['transfer_cases']]

    def score(part, results):
        assert researched == ['original', 'revised', 'random_choice', 'variance_choice']
        scored.append(results[0][1]['branch'])
        return [{'case': case['id'], 'kind': case['kind'], 'status': 'method_stopped',
                 'costs': result['costs'], 'terminal': {'passed': False}}
                for case, result in results]

    def forbidden(*args, **kwargs):
        raise AssertionError('Target continuation must not call a provider or restart the author.')

    monkeypatch.setattr(revision, 'frozen_transfers', targets)
    monkeypatch.setattr(revision, 'score_frozen', score)
    monkeypatch.setattr(revision, 'CodexCodeProvider', forbidden)
    monkeypatch.setattr(ExperimentalDesignMethodLearner, 'run', forbidden)
    summary = revision.resume_revision_targets(output)
    assert {name: len(rows) for name, rows in summary['frozen'].items()} == {
        'original': 10, 'revised': 22, 'random_choice': 12, 'variance_choice': 12}
    assert summary['unavailable_target_episodes'] == 0
    assert scored == researched
    assert summary['restored_practice_trials'] == 36
    assert len(summary['pairs']) == 10
    assert len(summary['choice_pairs']) == len(summary['variance_pairs']) == 12


def test_target_entry_rejects_unpublished_author_without_substitution(tmp_path):
    write_json(tmp_path / 'protocol.json', {})
    write_json(tmp_path / 'author/training.json', {'status': 'token_budget_exhausted',
                                                'method_source': None})
    write_json(tmp_path / 'published_method.json', {'published': False, 'source': None})
    with pytest.raises(ValueError, match='actual published source'):
        revision.resume_revision_targets(tmp_path)


def test_target_cli_never_initializes_subscription_or_author(monkeypatch, tmp_path):
    monkeypatch.setattr(revision.sys, 'argv', ['revision', '--resume-targets', str(tmp_path)])
    seen = []
    monkeypatch.setattr(revision, 'resume_revision_targets', lambda path: seen.append(path) or {
        'method_published': True, 'source_changed': True, 'new_costs': {}})
    monkeypatch.setattr(revision, 'verify_codex_subscription',
        lambda: pytest.fail('The frozen target entry has no subscription dependency.'))
    revision.main()
    assert seen == [tmp_path]


def test_pending_submission_resumes_its_confirmation_without_reinvoking_method(
        tmp_path, prospective_source):
    plan = revision.protocol(prospective_source, conditional_design=True)
    case = plan['transfer_cases'][10]
    plan['transfer_cases'] = [case]
    knowledge = read_json(prospective_source / 'initial_archive.json')
    predictor = "def fit(rows):\n    global labels\n    labels = {str(row['features']): row['target'] for row in rows}\ndef predict(row):\n    return labels.get(str(row['features']), 0.0)\n"
    method = '''import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    if not steps:
        return {'action': 'write', 'arguments': {'path': 'local.py',
            'code': __PREDICTOR__}}
    if len(steps) == 1:
        return {'model': 'local.py'}
    raise RuntimeError('The submitted explanation must not be chosen again.')
'''.replace('__PREDICTOR__', repr(predictor))
    output = tmp_path / 'pending'
    path = output / 'frozen' / case['id']
    prepare = learning.trial_preparer(knowledge, [], output / 'reference_archive',
                                     agent_type=ExperimentalDesignMethodAgent)
    initial = prepare(case, path, method, InquiryLLMConfig(**plan['limits']))

    def interrupt_at_submission(result):
        write_json(path / 'active.json', result)
        if result['method_results'][-1]['feedback'].get('status') == 'confirmation_not_authorized':
            raise InterruptedError('The declaration is saved; confirmation has not executed.')

    with pytest.raises(InterruptedError):
        initial['agent'].execute_method(initial['laboratory'], max_steps=32,
                                       on_checkpoint=interrupt_at_submission)
    saved = read_json(path / 'active.json')
    assert len(saved['method_results']) == 2
    assert saved['costs']['observation_cost'] == 72
    results = learning.frozen_transfers(plan, output, knowledge, [], method,
                                       agent_type=ExperimentalDesignMethodAgent)
    result = results[0][1]
    assert result['status'] == 'confirmation_failed'
    assert len(result['method_results']) == 2
    assert result['costs']['observation_cost'] == 168
    assert result['fresh_confirmation']['rows'] == 96
    assert 'RuntimeError' not in json.dumps(result['method_results'])


def test_restored_random_control_continues_same_draw_stream(tmp_path, monkeypatch):
    from random import Random
    from scripts.experimental_design_control import RandomExperimentMethodAgent
    from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
    comparisons = []
    rng = Random(9173 + 104729)
    for choices in (['a', 'b'], ['c', 'd', 'e'], ['a', 'b']):
        selected = rng.choice(choices)
        comparisons.append({'experiment_selection_control': {
            'policy': 'uniform_random_same_observation_cost',
            'eligible_experiments': choices, 'selected_experiment': selected}})

    def restore(agent, *args, **kwargs):
        agent.experiment_comparisons = deepcopy(comparisons)
        return {'restored_paid_rows': 72}

    monkeypatch.setattr(CommonArchiveInquiryAgent, 'restore_practice_research', restore)
    agent = RandomExperimentMethodAgent(None, tmp_path / 'control', arm='hegel',
        knowledge={'files': {}, 'lessons': []}, config=InquiryLLMConfig(), source_task=False)
    agent.restore_practice_research({}, tmp_path / 'old', None, confirmation_seed=9173)
    assert agent._experiment_rng.getstate() == rng.getstate()
