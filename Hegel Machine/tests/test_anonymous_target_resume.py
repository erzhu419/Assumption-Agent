"""Resume paid research without reauthoring, repurchasing prefixes, or leaking source fits."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.code_agent import evaluate_python_model, retain_evaluated_python_model
from hegel_machine.cognition.code_provider_context import _file_contents_preview
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
    author = SimpleNamespace(workspace=workspace, files={}, knowledge=deepcopy(knowledge),
                             common_interface='existing publication requirements')
    evidence = revision.attach_online_research_experience(author, source)
    assert evidence['fresh_confirmation']['passed']
    assert (workspace / evidence['model']).read_text() == (source / 'shared/model.py').read_text()
    assert (workspace / 'online_research_experience/shared').is_symlink()
    assert not (workspace / 'online_research_experience/evaluation.json').exists()
    assert 'terminal_marker' not in json.dumps(author.files)
    catalog = json.loads(author.files['source_catalog.json'])
    assert knowledge == before
    assert author.knowledge == {**before, 'source_catalog': catalog,
        'files': {**before['files'], 'source_catalog.json': author.files['source_catalog.json']}}
    assert set(author.files) == {'online_research_experience.json', 'source_catalog.json'}
    assert 'existing publication requirements' in author.common_interface


def test_online_experience_replays_submitted_context_with_root_parameters_and_fitting_rows(tmp_path):
    source = tmp_path / 'online'
    rows = [{'features': {'x': x}, 'target': 2 * x + 3} for x in (1., 2.)]
    write_json(source / 'model_observations.json', rows)
    write_json(source / 'parameters.json', {'offset': 3.})
    (source / 'model.py').write_text('''import json
def fit(rows):
    global coefficient, offset
    offset = json.load(open('parameters.json'))['offset']
    coefficient = sum((row['target'] - offset) / row['features']['x']
                      for row in rows) / len(rows)
def predict(row):
    return coefficient * row['features']['x'] + offset
''')
    write_json(source / 'evaluation.json', {'terminal_marker': 'not author input'})
    destination = Path('_submitted_models/confirmation_0000/_evaluated_model')
    retained = retain_evaluated_python_model(source, 'model.py', [], 0., destination=destination)
    submitted = {'model': str(destination / retained['model']),
                 'source_model': 'model.py', 'paid_rows': 2, 'mse': 0.}
    result = {'status': 'completed', 'model': 'model.py', 'costs': {'observation_cost': 98},
              'fresh_confirmation': {'passed': True, 'mse': 0., 'submitted_candidate': submitted}}
    write_json(source / 'training.json', result)
    # Continuing edits do not redefine the already submitted predictor.
    write_json(source / 'parameters.json', {'offset': 100.})
    (source / 'model.py').write_text('def predict(row): return -1.')
    original_files = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}

    workspace = tmp_path / 'author'
    write_json(workspace / 'parameters.json', {'offset': -999.})
    write_json(workspace / 'model_observations.json', [])
    knowledge = {'files': {'old.py': 'untouched'}, 'source_catalog': []}
    author = SimpleNamespace(workspace=workspace, files={}, knowledge=deepcopy(knowledge),
                             common_interface='existing publication requirements')
    evidence = revision.attach_online_research_experience(author, source)
    candidate = evidence['confirmed_candidate']
    context = workspace / candidate['working_directory']
    assert context.is_dir() and not context.is_symlink()
    # This is the exact inspection used by the real method author. A symlink
    # at _evaluated_model made it return no Python source and led to inspecting
    # an unrelated archived predictor instead of this confirmed one.
    discovered = list((workspace / 'online_research_experience/confirmed_candidate').rglob('*.py'))
    assert workspace / candidate['model'] in discovered
    assert (workspace / candidate['model']).read_text() == (source / destination / 'model.py').read_text()
    assert read_json(context / 'parameters.json') == {'offset': 3.}
    assert read_json(context / 'model_observations.json') == rows
    replay = evaluate_python_model(workspace, candidate['model'],
                                   [{'features': {'x': 4.}, 'target': 11.}], require_fit=True)
    assert replay['returncode'] == 0, replay
    assert replay['mse'] == pytest.approx(0.)
    assert replay['predictions'] == [11.]
    assert candidate['source_model'] == 'model.py'
    assert evidence['costs'] == result['costs']
    catalog = json.loads(author.files['source_catalog.json'])
    assert author.knowledge == {**knowledge, 'source_catalog': catalog,
        'files': {**knowledge['files'], 'source_catalog.json': author.files['source_catalog.json']}}
    assert set(author.files) == {'online_research_experience.json', 'source_catalog.json'}
    assert not (context / 'evaluation.json').exists()
    assert read_json(source / 'training.json') == result
    assert all(path.read_bytes() == contents for path, contents in original_files.items())
    # The copied execution inputs can be inspected or replayed without making
    # later author edits mutate the paid submitted snapshot.
    write_json(context / 'parameters.json', {'offset': -12.})
    assert read_json(source / destination / 'parameters.json') == {'offset': 3.}


def test_online_experience_distinguishes_failed_shared_procedure_from_confirmed_online_predictor(tmp_path):
    source, workspace = tmp_path / 'online', tmp_path / 'author'
    events = [{'step': i, 'action': 'run', 'arguments': {'code': 'retained inspection'},
               'feedback': {'returncode': 0, 'stdout': 'large output' * 10000}} for i in range(20)]
    events[-2] = {'step': 18, 'action': 'evaluate', 'arguments': {'model': 'wrong_rewrite.py'},
                  'feedback': {'returncode': 0, 'mse': 2.08}}
    events[-1] = {'step': 19, 'action': 'finish', 'arguments': {'model': 'native_candidate.py'},
                  'feedback': {'returncode': 0, 'mse': 1e-23,
                               'fresh_confirmation': {'passed': True, 'mse': 2e-23, 'rows': 96}}}
    write_json(source / 'training.json', {'status': 'completed', 'model': 'native_candidate.py',
        'costs': {'observation_cost': 7901}, 'events': events,
        'fresh_confirmation': {'passed': True, 'mse': 2e-23, 'rows': 96,
                               'residuals': {'source_only_history': 'x' * 400000}}})
    (source / 'native_candidate.py').write_text('def predict(row): return 0.0\n')
    write_json(source / 'shared/knowledge.json', {'prior_execution': 'older successful procedure'})
    (source / 'shared/consolidated_method.py').write_text(
        "def select_actions(spec, records, memory): return {'stop': 'unresolved old rewrite'}\n")
    decision = {'methods': [{'id': 'old_method', 'path': 'shared/consolidated_method.py',
        'source_unchanged': False,
        'prior_execution': {'scope': 'Earlier source practice, not current-task support.',
            'cases': [{'case': 'old_case', 'status': 'completed',
                       'fresh_confirmation': {'passed': True, 'mse': 0.0, 'rows': 96}}]},
        'current_executions': [{'step': 0, 'proposal': {'stop': 'unresolved old rewrite'},
                                'feedback': {'status': 'method_stopped'}}],
        'continuation': {'status': 'method_stopped'}}],
        'research_options': [{'id': 'native_condition_research', 'current_executions': [{
            'question_id': 'research', 'round_index': i, 'status': 'candidates_require_evaluation',
            'paid_rows': 7805,
            'models': [{'model': f'_experiment_models/round_{i}/candidate_{j}.py', 'fit_mse': 1e-23,
                        'evaluate_action': {'action': 'evaluate', 'arguments': {
                            'model': f'_experiment_models/round_{i}/candidate_{j}.py'}}}
                       for j in range(12)],
            'source': {'file': 'research_state.json', 'json_path': ['questions', 0, 'native_condition_research', i]}}
            for i in range(7)]}]}
    for item in decision['research_options'][0]['current_executions']:
        for model in item['models']:
            path = source / model['model']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('def predict(row): return 0.0\n')
    write_json(source / 'method_decision.json', decision)
    write_json(source / 'research_state.json', {'method_results': decision['methods'][0]['current_executions']})
    result = read_json(source / 'training.json')
    result['method_results'] = [{**deepcopy(step), 'path': 'shared/consolidated_method.py'}
        for step in decision['methods'][0]['current_executions']]
    write_json(source / 'training.json', result)
    original = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}
    workspace.mkdir()
    author = SimpleNamespace(workspace=workspace, files={}, knowledge={'files': {'old.py': 'unchanged'}},
                             common_interface='existing publication requirements')
    knowledge = deepcopy(author.knowledge)
    evidence = revision.attach_online_research_experience(author, source)
    execution = evidence['source_execution']
    old_method = execution['methods'][0]
    assert old_method['prior_execution']['cases'][0]['fresh_confirmation']['passed']
    assert old_method['source_unchanged'] is False
    assert old_method['execution_summary']['status'] == 'method_stopped'
    native = execution['native_research'][0]
    assert native['execution_count'] == 7
    assert [item['round_index'] for item in native['recent_executions']] == [4, 5, 6]
    assert native['recent_executions'][-1]['model_preview'][0]['evaluate_action']['arguments']['model'] == (
        '_experiment_models/round_6/candidate_0.py')
    for item in native['recent_executions']:
        for model in item['model_preview']:
            path = workspace / execution['working_directory'] / model['model']
            assert path.is_file() and path.resolve() == source / model['model']
    assert (workspace / execution['working_directory'] / '_experiment_models').is_symlink()
    assert 'online_research_experience/_experiment_models' in evidence['files']
    assert execution['online_preview_start'] == 8
    assert [event['event_index'] for event in execution['recent_online_actions']] == list(range(8, 20))
    assert execution['recent_online_actions'][-2]['feedback']['mse'] == 2.08
    assert execution['recent_online_actions'][-1]['feedback']['fresh_confirmation']['passed']
    assert execution['recent_online_actions'][-1]['arguments']['model'] == 'native_candidate.py'
    linked_decision = workspace / execution['working_directory'] / execution['method_decision_file']
    assert linked_decision.is_symlink() and linked_decision.resolve() == source / 'method_decision.json'
    assert 'method_results' in execution['trace_fields']['procedure_steps']
    assert 'events' in execution['trace_fields']['online_actions']
    assert evidence['fresh_confirmation'] == {'passed': True, 'mse': 2e-23, 'rows': 96}
    rendered = author.files['online_research_experience.json']
    assert len(rendered) < 12000
    visible = _file_contents_preview(author.files, ['files'])['online_research_experience.json']
    assert isinstance(visible, str) and visible == rendered
    assert json.loads(visible)['source_execution'] == execution
    catalog = json.loads(author.files['source_catalog.json'])
    assert set(author.files) == {'online_research_experience.json', 'source_catalog.json'}
    assert author.knowledge == {**knowledge, 'source_catalog': catalog,
        'files': {**knowledge['files'], 'source_catalog.json': author.files['source_catalog.json']}}
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize('decision_kind', ['structural_mismatch', 'remaining_parameter_ambiguity'])
def test_online_experience_uses_canonical_method_steps_after_execution_view_compaction(
        tmp_path, decision_kind):
    source, workspace = tmp_path / 'online', tmp_path / 'author'
    method = 'shared/consolidated_method.py'
    model = '_experiment_models/round_0/candidate.py'
    next_query = {'action': 'query', 'arguments': {
        'actions': [{'kind': 'trajectory', 'channels': [], 'sequence': [{'u': .183}]}],
        'after_observation': [{'operator': 'native_condition_research'}]}}
    native = {'status': 'candidates_require_evaluation', 'round_index': 2,
        'candidates': [{'model': model, 'fit_mse': .02,
            'evaluate_action': {'action': 'evaluate', 'arguments': {'model': model}}}],
        'research_decision': {'kind': decision_kind, 'next_experiment': next_query},
        'remaining_boundary': {'threshold_interval': [.17, .19], 'interval_width': .02},
        'proposed_experiments': [{'purpose': 'conditional_boundary',
            'observation_cost': 1, 'query_action': next_query}]}
    steps = [
        {'path': 'shared/other.py', 'proposal': {'model': 'other.py'},
         'feedback': {'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96}}},
        {'step': 0, 'path': method, 'proposal': {'action': 'query',
             'arguments': {'actions': next_query['arguments']['actions']}},
         'feedback': {'after_observation_results': [{'operator': 'native_condition_research',
             'native_condition_research': native}]}},
        {'step': 1, 'path': method, 'proposal': {'stop': 'paid structural counterexample'},
         'feedback': {'status': 'method_stopped'}},
        {'path': 'shared/other.py', 'proposal': {'model': 'other.py'},
         'feedback': {'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96}}},
    ]
    result = {'status': 'completed', 'model': 'final.py', 'method_results': steps,
        'costs': {'observation_cost': 204}, 'fresh_confirmation': {'passed': True, 'mse': 0.}}
    write_json(source / 'training.json', result)
    write_json(source / 'research_state.json', {'method_results': steps})
    (source / 'final.py').write_text('def predict(row): return 0.\n')
    path = source / model
    path.parent.mkdir(parents=True)
    path.write_text('def predict(row): return 0.\n')
    compact_steps = deepcopy(steps[1:3])
    compact_steps[0]['feedback']['after_observation_results'] = {
        'full_content_file': 'research_state.json',
        'json_path': ['method_results', 1, 'feedback', 'after_observation_results']}
    write_json(source / 'method_decision.json', {'methods': [{
        'id': 'retained', 'path': method, 'source_unchanged': True,
        'current_executions': compact_steps}], 'research_options': []})
    workspace.mkdir()
    author = SimpleNamespace(workspace=workspace, files={}, knowledge={'files': {}},
                             common_interface='existing publication rules')
    evidence = revision.attach_online_research_experience(author, source)
    execution = evidence['source_execution']
    summary, = [item['execution_summary'] for item in execution['methods']]
    assert summary['status'] == 'method_stopped'
    assert summary['step_count'] == 2
    assert summary['submission_count'] == summary['confirmation_count'] == 0
    assert summary['result_file'] == 'training.json'
    assert summary['result_steps_field'] == 'method_results'
    returned = summary['latest_native_result']
    assert returned['candidate_count'] == 1
    assert returned['candidate_preview'][0]['evaluate_action'] == native['candidates'][0]['evaluate_action']
    assert returned['research_decision'] == native['research_decision']
    assert returned['remaining_boundary'] == native['remaining_boundary']
    assert returned['json_path'] == ['method_results', 1, 'feedback',
        'after_observation_results', 0, 'native_condition_research']
    resolved = read_json(workspace / execution['working_directory'] / summary['result_file'])
    for key in returned['json_path']:
        resolved = resolved[key]
    assert resolved == native
    proposal = result
    for key in returned['proposed_experiment_preview'][0]['query_action_path']:
        proposal = proposal[key]
    assert proposal == next_query
    visible = _file_contents_preview(author.files, ['files'])['online_research_experience.json']
    assert json.loads(visible)['source_execution']['methods'][0]['execution_summary'] == summary
    assert read_json(source / 'training.json') == result


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
