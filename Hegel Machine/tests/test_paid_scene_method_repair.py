"""Exercise the author edit/practice connection on an actual retained paid scene."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.anonymous_method_learning import (
    EMPTY_METHOD, SourceProcedureRepairLearner, paid_scene_trial_preparer,
)
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.code_agent import archived_paid_trace_references
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_native_condition_inquiry import execute
from test_temporal_python_refit import LawLab, MODEL, DEPENDENCY


def make_repair(tmp_path, *, extra_files=None, reference_archive=None):
    knowledge = {'files': {METHOD_PATH: EMPTY_METHOD, 'shared/knowledge.json':
        json.dumps({'hypotheses': [], 'research_programs': [], 'support_questions': []})}}
    knowledge['files'].update(extra_files or {})
    lab = LawLab(slope=-2., offset=.4)
    config = InquiryLLMConfig(observation_budget=192, confirmation_rows=12,
        execution_seconds=60., compute_budget=10000, max_model_calls=0, token_budget=0)
    source = CommonArchiveInquiryAgent(None, tmp_path / 'source', arm='hegel',
        source_task=False, knowledge=deepcopy(knowledge), config=config, confirmation_seed=173)
    rng = Random(93)
    execute(source, lab, 'query', {'actions': [
        {'kind': 'trajectory', 'channels': [],
         'sequence': [{'u': rng.uniform(-1., 1.)} for _ in range(12)]} for _ in range(3)]})
    for path, text in [('model.py', MODEL), ('fitting_dependency.py', DEPENDENCY),
                       ('settings.json', json.dumps({'basis': 'linear'}))]:
        source._save(path, text)
    values = {json.dumps([dict(row.features), list(row.history)], sort_keys=True): row.target
              for row in source.rows}
    source._save('memorized.py', 'import json\n' + f'VALUES = {values!r}\n' +
        "def predict(row):\n    return VALUES.get(json.dumps([row['features'], row.get('history', [])], sort_keys=True), 0.)\n")
    rejected = execute(source, lab, 'finish', {'model': 'memorized.py'})
    assert rejected['fresh_confirmation']['passed'] is False
    source._prepare_program_state()
    previous = source.result('model_call_budget_exhausted')
    checkpoint = source.workspace / 'training.json'
    checkpoint.write_text(json.dumps(previous))
    previous = json.loads(checkpoint.read_text())
    (source.workspace / 'research_context.json').write_text(json.dumps(source._request(lab)))
    paid_traces = (archived_paid_trace_references(knowledge['files'], reference_archive,
        source_workspace=source.workspace) if reference_archive is not None else None)
    author = SourceProcedureRepairLearner(None, tmp_path / 'author', knowledge=knowledge,
        cases={'same_scene': {'confirmation_seed': 173}},
        lab_factory=lambda _: LawLab(slope=-2., offset=.4), config=config,
        practice_limits=config, paid_trace_references=paid_traces,
        prepare_trial=paid_scene_trial_preparer(knowledge, checkpoint,
            lambda _: LawLab(slope=-2., offset=.4), paid_trace_references=paid_traces))
    (author.workspace / 'scene_source').symlink_to(source.workspace, target_is_directory=True)
    author.files['scene_source.json'] = json.dumps({'working_directory': 'scene_source',
        'training': 'scene_source/training.json', 'program_state': 'scene_source/research_state.json',
        'observations': 'scene_source/laboratory_observations.json'})
    (author.workspace / 'scene_source.json').write_text(author.files['scene_source.json'])
    return author, checkpoint, previous


def test_paid_scene_author_and_repeated_practice_link_original_traces_without_rewriting(tmp_path, monkeypatch):
    from pathlib import Path

    trace = 'archive/source_000/paid_trace.json'
    repeated = 'archive/source_001/_evaluated_model/archive/source_000/paid_trace.json'
    content = '{"paid_evidence": "original source"}'
    author, checkpoint, previous = make_repair(tmp_path,
        extra_files={trace: content, repeated: content}, reference_archive=tmp_path / 'reference_archive')
    original = checkpoint.parent / trace
    original_bytes = original.read_bytes()
    assert (tmp_path / 'reference_archive' / trace).is_symlink()
    assert (author.workspace / trace).is_symlink()
    assert (author.workspace / trace).resolve() == original.resolve()
    assert (author.workspace / repeated).resolve() == original.resolve()
    write_text = Path.write_text

    def no_trace_write(path, *args, **kwargs):
        assert path.name != 'paid_trace.json', 'paid scene reconstructed immutable trace text'
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', no_trace_write)
    for _ in range(2):
        trial = author._practice('same_scene')
        directory = author.workspace / trial['directory']
        assert (directory / trace).is_symlink()
        assert (directory / trace).resolve() == original.resolve()
        assert trial['continuation_cost']['observation_cost'] == 0
        saved = json.loads((directory / 'result.json').read_text())
        assert saved['costs']['observation_cost'] == previous['costs']['observation_cost']
        assert saved['knowledge_protocol'] == 'method_practice_workspace'
        assert 'files' not in saved['outgoing_knowledge']
    assert original.read_bytes() == original_bytes


# This is a test-only affine fitting program, never supplied to the real author.
REFIT_METHOD = '''import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    last = steps[-1]['feedback'] if steps else {}
    if last.get('operator') == 'refit':
        return {'action': 'evaluate', 'arguments': {'model': last['path']}}
    if steps and steps[-1]['proposal'].get('action') == 'evaluate':
        return {'model': steps[-1]['proposal']['arguments']['model']}
    return {'action': 'run', 'arguments': {'operator': 'refit', 'model': 'model.py',
        'path': 'shared/current_refit', 'validation_records': [2]}}
'''


def test_paid_scene_edit_runs_complete_program_and_confirms_without_publishing(tmp_path):
    author, checkpoint, previous = make_repair(tmp_path)
    author.config = replace(author.config, confirmation_rows=0)
    request = author._request(author.interface)
    assert request['limits']['confirmation_rows'] == 0
    assert request['practice_confirmation'] == {
        'confirmation_rows': 12,
        'confirmation_mse_threshold': author.practice_limits.confirmation_mse_threshold,
        'remaining_observations': 192,
        'remaining_compute': 10000,
        'remaining_execution_seconds': 60.,
        'resources_allow_practice': True,
    }
    old_bytes = checkpoint.read_bytes()
    stopped = author._practice('same_scene')
    assert stopped['status'] == 'method_stopped'
    assert stopped['fresh_confirmation'] is None
    assert stopped['continuation_cost']['observation_cost'] == 0
    assert 'error' in author._execute('finish', {'path': METHOD_PATH}, author.interface)
    edited = author._execute('write', {'path': METHOD_PATH, 'code': REFIT_METHOD}, author.interface)
    practice = edited['edit_practice']
    assert practice['status'] == 'completed', practice
    assert practice['fresh_confirmation']['passed'] is True
    assert practice['execution_summary']['step_count'] == 3
    assert practice['execution_summary']['confirmation_count'] == 1
    assert practice['continuation_cost']['observation_cost'] == 12
    assert practice['costs']['observation_cost'] == previous['costs']['observation_cost'] + 12
    assert author.costs['observation_cost'] == 12
    saved = json.loads((author.workspace / practice['directory'] / 'result.json').read_text())
    assert saved['paid_observations'][:len(previous['paid_observations'])] == previous['paid_observations']
    assert saved['resume_origin'].endswith('trial_000/result.json')
    request = author._request(author.interface)
    account = request['practice_confirmation']
    assert account['remaining_observations'] == 180
    assert account['remaining_compute'] == 10000 - author.costs['compute_cost']
    assert account['remaining_execution_seconds'] == 60. - author.costs['python_execution_seconds']
    assert account['resources_allow_practice'] is True
    current = request['paid_scene']
    assert current['working_directory'] == practice['directory']
    assert current['initial_scene']['working_directory'] == 'scene_source'
    assert current['costs']['observation_cost'] == previous['costs']['observation_cost'] + 12
    assert current['fresh_confirmation']['passed'] is True
    assert json.loads((author.workspace / current['training']).read_text()) == saved
    # Successful confirmation stays outside the predictor's fitting files;
    # result.json is the complete paid record, including that final batch.
    assert json.loads((author.workspace / current['observations']).read_text()) == previous['paid_observations']
    finished = author._execute('finish', {'path': METHOD_PATH, 'lesson': 'fixture fit checked'}, author.interface)
    assert finished['status'] == 'source_repair_confirmed'
    result = author.result('completed')
    assert result['status'] == 'source_repair_confirmed'
    assert result['published'] is False
    assert result['method_source'] is None and result['model'] is None
    assert author.published_source is None
    assert checkpoint.read_bytes() == old_bytes


def test_paid_scene_confirmation_request_reports_insufficient_shared_observations(tmp_path):
    author, checkpoint, _ = make_repair(tmp_path)
    author.config = replace(author.config, confirmation_rows=0, observation_budget=11)
    before = checkpoint.read_bytes()
    request = author._request(author.interface)
    account = request['practice_confirmation']
    assert account['confirmation_rows'] == 12
    assert account['remaining_observations'] == 11
    assert account['resources_allow_practice'] is False
    assert 'author does not directly submit predictors' in request['task_interface']
    assert author._practice('same_scene') == {
        'error': 'remaining resources cannot cover fresh confirmation'}
    assert author.practice_trials == []
    assert author.costs['observation_cost'] == 0
    assert checkpoint.read_bytes() == before


def test_new_draft_cannot_claim_previous_confirmation_and_contract_uses_paid_scene(tmp_path):
    author, _, _ = make_repair(tmp_path)
    assert author._execute('write', {'path': METHOD_PATH, 'code': REFIT_METHOD}, author.interface)[
        'edit_practice']['fresh_confirmation']['passed'] is True
    stopped = author._execute('write', {'path': METHOD_PATH, 'code': EMPTY_METHOD}, author.interface)
    assert stopped['edit_practice']['fresh_confirmation'] is None
    assert stopped['edit_practice']['execution_summary']['confirmation_count'] == 0
    explained = EMPTY_METHOD.replace('No research method has been authored yet.',
        'The earlier confirmation does not confirm this stopped method.')
    edited = author._execute('write', {'path': METHOD_PATH, 'code': explained}, author.interface)
    assert edited['edit_practice_skipped']['reason'] == 'stop_explanation_only'
    assert len(author.practice_trials) == 2
    assert edited['current_source_coverage']['missing_cases'] == ['same_scene']
    assert 'error' in author._execute('finish', {'path': METHOD_PATH}, author.interface)
    assert author.model is None
    contract = author._request(author.interface)['task_interface']
    index = json.loads((author.workspace / 'scene_source.json').read_text())
    assert index['working_directory'] == stopped['edit_practice']['directory']
    assert index['initial_scene']['working_directory'] == 'scene_source'
    assert index['fresh_confirmation'] is None
    assert 'Practices continue the latest paid scene' in contract
    assert 'fresh paid 72-row prefix' not in contract
    assert 'initial method is only an unresolved placeholder' not in contract
    assert '12 fresh rows' in contract


def test_stop_explanation_edits_save_source_without_new_trial_and_allow_explicit_practice(tmp_path):
    author, _, _ = make_repair(tmp_path)
    stopped = author._practice('same_scene')
    assert stopped['status'] == 'method_stopped'
    before_costs = deepcopy(author.costs)
    before_trials = deepcopy(author.practice_trials)
    for message in ['The candidate needs more evidence.', 'Its validation remains unresolved.']:
        source = EMPTY_METHOD.replace('No research method has been authored yet.', message)
        feedback = author._execute('write', {'path': METHOD_PATH, 'code': source}, author.interface)
        assert 'edit_practice' not in feedback
        assert feedback['edit_practice_skipped']['previous_practice'] == stopped['directory']
        assert author.files[METHOD_PATH] == source
        assert (author.workspace / METHOD_PATH).read_text() == source
        assert author.costs == before_costs
        assert author.practice_trials == before_trials
        assert list((author.workspace / 'trials').glob('trial_*')) == [
            author.workspace / stopped['directory']]
    final_source = source.replace('Its validation remains unresolved.', 'Preserve the unresolved evidence.')
    feedback = author._execute('run', {'code':
        f'from pathlib import Path\nPath({METHOD_PATH!r}).write_text({final_source!r})'}, author.interface)
    assert feedback['edit_practice_skipped']['reason'] == 'stop_explanation_only'
    assert 'edit_practice' not in feedback
    assert author.practice_trials == before_trials
    assert (author.workspace / METHOD_PATH).read_text() == final_source
    assert feedback['current_source_coverage']['missing_cases'] == ['same_scene']
    action = feedback['edit_practice_skipped']['practice_action']
    explicit = author._execute(action['action'], action['arguments'], author.interface)
    assert explicit['status'] == 'method_stopped'
    assert explicit['directory'] != stopped['directory']
    assert len(author.practice_trials) == 2
    assert author.practice_trials[-1]['method_source'] == final_source
    assert 'error' in author._execute('finish', {'path': METHOD_PATH}, author.interface)


@pytest.mark.parametrize(('old', 'new'), [
    ("'action': 'evaluate'", "'action': 'run'"),
    ("'model': 'model.py'", "'model': 'memorized.py'"),
    ('len(records) > 1000', 'len(records) > 2000'),
])
def test_stopped_method_action_parameter_and_branch_edits_still_practice(tmp_path, old, new):
    author, _, _ = make_repair(tmp_path)
    source = '''def select_actions(spec, records, memory):
    if len(records) > 1000:
        return {'action': 'evaluate', 'arguments': {'model': 'model.py'}}
    return {'stop': 'Current evidence is insufficient.'}
'''
    first = author._execute('write', {'path': METHOD_PATH, 'code': source}, author.interface)
    assert first['edit_practice']['status'] == 'method_stopped'
    changed = source.replace(old, new).replace('Current evidence is insufficient.', 'Still unresolved.')
    feedback = author._execute('write', {'path': METHOD_PATH, 'code': changed}, author.interface)
    assert 'edit_practice_skipped' not in feedback
    assert feedback['edit_practice']['status'] == 'method_stopped'
    assert len(author.practice_trials) == 2
    assert author.practice_trials[-1]['method_source'] == changed


def test_paid_scene_separates_current_evaluation_from_prior_confirmation(tmp_path, monkeypatch):
    author, _, _ = make_repair(tmp_path)
    confirmed = author._execute('write', {'path': METHOD_PATH, 'code': REFIT_METHOD}, author.interface)[
        'edit_practice']
    assert confirmed['fresh_confirmation']['passed'] is True
    refit_then_stop = REFIT_METHOD.replace(
        "return {'model': steps[-1]['proposal']['arguments']['model']}",
        "return {'stop': 'The current refit remains unsubmitted.'}").replace(
            'shared/current_refit', 'shared/latest_refit')
    current = author._execute('write', {'path': METHOD_PATH, 'code': refit_then_stop}, author.interface)[
        'edit_practice']
    assert current['status'] == 'method_stopped'
    result_path = author.workspace / current['directory'] / 'result.json'
    raw_result = result_path.read_bytes()
    # The next request may read this new completed result once. Further author
    # reads reuse that small metadata rather than reparsing the paid history.
    author._paid_scene_scores = None
    reads = []
    read_text = type(result_path).read_text

    def counted_read(path, *args, **kwargs):
        if path == result_path:
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(result_path), 'read_text', counted_read)
    before = deepcopy(author.costs)
    for _ in range(2):
        request = author._request(author.interface)
        scene = request['paid_scene']
        assert scene['fresh_confirmation'] is None
        retained = scene['evaluated_model']
        submitted = scene['latest_submitted_predictor']
        assert retained['evaluated_rows'] == current['costs']['observation_cost']
        recorded = json.loads(raw_result)['evaluated_model']
        assert {k: v for k, v in retained.items() if k != 'model_file'} == recorded
        assert submitted['fresh_confirmation']['passed'] is True
        assert submitted['fresh_confirmation']['rows'] == 12
        assert submitted['paid_rows'] + 12 == retained['evaluated_rows']
        assert submitted['model_file'] != retained['model_file']
        assert (author.workspace / submitted['model_file']).is_file()
        assert (author.workspace / retained['model_file']).is_file()
        assert submitted['json_path'][0] == 'method_results'
        assert json.loads(request['files']['scene_source.json']) == scene
    assert reads == [result_path]
    assert author.costs == before
    assert result_path.read_bytes() == raw_result
