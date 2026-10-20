"""The author sees executed continuations, without copying their large payloads."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.anonymous_method_learning import (
    EMPTY_METHOD, SourceProcedureRepairLearner, paid_scene_trial_preparer,
)
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_evidence_procedure_execution import PARENT
from test_executable_evidence_binding import return_local_checks, return_region
from test_inquiry_revision import action
from test_temporal_python_refit import LawLab


CHILD = 'shared/continuation.py'


def laboratory(_):
    return LawLab(slope=1., offset=0.)


def continuation_source(path, revision):
    # Toy fitting source and checked child results come from existing fixtures;
    # this program tests only the supported execution/feedback connection.
    return f'''import json
def select_actions(spec, records, memory):
    state = json.load(open('research_state.json'))
    own = [s for s in state['method_results'] if s['path'] == {path!r}]
    if not own:
        return {{'actions': [], 'after_observation': [
            {{'operator':'bind_research', 'model':'parent.py', 'inputs':{{
                'slope':'slope_question', 'contrast':{{'revision_index':{revision}}}}}}},
            {{'operator':'refit', 'model':'parent.py', 'validation_records':[2],
                'path':'shared/continued_fit'}}]}}
    if len(own) == 1:
        candidate = own[-1]['feedback']['after_observation_results'][-1]['path']
        return {{'action':'evaluate', 'arguments':{{'model':candidate}}}}
    return {{'stop':'Return the executed evidence and candidate comparison.'}}
'''


@pytest.mark.parametrize('nested', [False, True])
def test_source_work_reports_actual_continuation_refit_and_bindings_once(tmp_path, nested):
    lab = laboratory(None)
    scene = CommonArchiveInquiryAgent(None, tmp_path / 'scene', arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=30, confirmation_rows=3,
            execution_seconds=30., compute_budget=10000))
    scene._execute('query', {'actions': [action(-1., 1.), action(-.5, .5),
                                          action(.75, .25)]}, lab)
    scene._save('parent.py', PARENT)
    scene._execute('evaluate', {'model': 'incumbent.py',
        'code': "def predict(row): return row['features']['u']\n"}, lab)
    return_region(scene, lab, .9, 1.1)
    checked = return_local_checks(scene, lab, expected=2.)
    scene._save(METHOD_PATH, EMPTY_METHOD)
    scene._prepare_program_state()
    previous = scene.result('model_call_budget_exhausted')
    checkpoint = scene.workspace / 'training.json'
    checkpoint.write_text(json.dumps(previous))
    (scene.workspace / 'research_context.json').write_text(json.dumps(scene._request(lab)))
    original = checkpoint.read_bytes()
    knowledge = {'files': {METHOD_PATH: EMPTY_METHOD, 'shared/knowledge.json':
        json.dumps({'hypotheses': [], 'research_programs': [], 'support_questions': []})}}
    author = SourceProcedureRepairLearner(None, tmp_path / 'author', knowledge=knowledge,
        cases={'scene': {'confirmation_seed': 73}}, lab_factory=laboratory,
        config=scene.config, practice_limits=scene.config,
        prepare_trial=paid_scene_trial_preparer(knowledge, checkpoint, laboratory))
    direct = continuation_source(CHILD if nested else METHOD_PATH, checked['index'])
    source = direct
    if nested:
        source = f'''import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    own = [s for s in steps if s['path'] == {METHOD_PATH!r}]
    if not own:
        return {{'action':'write', 'arguments':{{'path':{CHILD!r}, 'code':{direct!r}}}}}
    if len(own) == 1:
        return {{'action':'run', 'arguments':{{'operator':'procedure', 'path':{CHILD!r}}}}}
    return {{'stop':'Return the completed child program to the author.'}}
'''
    selected = author.workspace / 'selected.py'
    selected.write_text(source)
    outcome = author._execute('run', {'operator': 'procedure', 'path': 'selected.py',
        'question': {'id': 'authored_work', 'question': 'Check my selected procedure'}}, author.interface)
    assert outcome['status'] == 'method_stopped', outcome
    operations = outcome['research_work_result']['operations']
    fits = [item for item in operations if item.get('operator') == 'refit']
    assert len(fits) == 1, operations
    fit = fits[0]
    assert fit['validation']['mse'] == 0.
    assert fit['split'] == {'train_rows': 4, 'holdout_rows': 2,
        'records_count': 3, 'training_records_count': 2, 'validation_records_count': 1}
    assert 'after_observation_results' in fit['evidence']['json_path']
    consumed = fit['executed_bindings'][0]
    assert consumed['actual_parameters'] == {'slope': 1.}
    assert consumed['inputs']['contrast']['revision_index'] == checked['index']
    assert consumed['inputs']['slope']['question_id'] == 'slope_question'
    saved = json.loads((author.workspace / fit['evidence']['file']).read_text())
    actual = deepcopy(saved)
    for key in fit['evidence']['json_path']:
        actual = actual[key]
    assert actual['validation']['mse'] == fit['validation']['mse']
    assert actual['validation']['research_bindings'][0]['actual_parameters'][
        'training_inputs'] == [-1., 1., -.5, .5]
    assert any(item.get('retention_decision') for item in operations)
    assert outcome['continuation_cost']['observation_cost'] == 0
    assert outcome['fresh_confirmation'] is None
    assert checkpoint.read_bytes() == original
    assert selected.read_text() == source
