"""Recover committed paid research when the compact checkpoint lags behind."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.common_archive_inquiry import (
    CommonArchiveInquiryAgent, retained_checkpoint_result,
)
from hegel_machine.cognition import registered_source_scene as scenes
from hegel_machine.cognition import registered_source_control as control
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.online_inquiry import InquiryLab


def costs(compute):
    return {'compute_cost': compute, 'python_execution_seconds': float(compute),
        'observation_cost': 0, 'measurement_cost': 0, 'model_calls': 0,
        'reported_tokens': 0}


def test_restore_adopts_CP8_paid_graph_without_rereading_or_using_CP7_questions(tmp_path, monkeypatch):
    source = CommonArchiveInquiryAgent(None, tmp_path / 'source', arm='code_react',
        config=InquiryLLMConfig())
    source.costs = costs(10)
    source.questions = [{'id': 'readout', 'meaning_version': 1, 'status': 'answered'},
        {'id': 'parent195', 'meaning_version': 1, 'status': 'open',
         'frozen_recall_execution': {'cursor': {'next_candidate_id': 'already_paid'}}}]
    source.events = [{'arguments': {'question': {'id': 'readout'}}, 'feedback': {}}]
    source._prepare_program_state()
    previous = deepcopy(source.checkpoint_result('CP7'))
    state = deepcopy(source._program_state())
    state['costs'] = costs(11)
    state['questions'][1]['frozen_recall_execution']['cursor']['next_candidate_id'] = 'next_unpaid'
    state['theory_index'] = {'paid7': {'id': 'paid7', 'version': 1}}
    state['operator_results'] = [{'index': 0, 'operator': 'research_choice',
        'arguments': {'operator': 'research_choice', 'models': ['paid7.py']}}]
    state['method_results'] = [{'path': 'paid7.py', 'step': 7, 'feedback': {}}]
    state['online_research']['active_question'] = 'parent195'
    state['revisions'] = [{'index': 0, 'id': 'seventh_paid_revision'}]
    (source.workspace / 'training.json.writing').write_text('uncommitted CP8 result')
    restored = CommonArchiveInquiryAgent(None, tmp_path / 'restored', arm='code_react',
        config=InquiryLLMConfig())
    original_read = Path.read_text

    def no_second_state_read(path, *args, **kwargs):
        if path == source.workspace / 'research_state.json':
            pytest.fail('the caller already decoded this committed state')
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', no_second_state_read)
    laboratory = InquiryLab(seed=8, mechanism='linear', driver=0, orientation=1)
    receipt = restored.restore_practice_research(previous, source.workspace, laboratory,
        confirmation_seed=9173, retained_state=state)
    assert restored.questions is state['questions']
    assert restored.theory_index is state['theory_index']
    assert restored.operator_results is state['operator_results']
    assert restored.method_results is state['method_results']
    assert restored.revisions is state['revisions']
    assert restored.questions[1]['frozen_recall_execution']['cursor'] == {
        'next_candidate_id': 'next_unpaid'}
    assert restored._active_question_id == 'parent195'
    assert restored.costs == state['costs'] == receipt['costs_before']
    assert receipt['new_observations'] == 0
    assert previous['costs'] == costs(10)
    assert previous['online_questions'][1]['frozen_recall_execution']['cursor'] == {
        'next_candidate_id': 'already_paid'}
    assert not (restored.workspace / 'training.json.writing').exists()


@pytest.mark.parametrize('field,value,error', [
    ('compute_cost', 9, 'account precedes'),
    ('observation_cost', 1, 'acquisitions absent'),
    ('model_calls', 1, 'acquisitions absent'),
])
def test_state_overlay_does_not_erase_fees_or_invent_paid_records(field, value, error):
    previous = {'costs': costs(10)}
    state = {'costs': costs(11), 'questions': [], 'revisions': []}
    state['costs'][field] = value
    with pytest.raises(ValueError, match=error):
        retained_checkpoint_result(previous, state)


def interrupted_material(tmp_path, monkeypatch):
    source = tmp_path / 'run' / 'remove'
    source.mkdir(parents=True)
    origin = tmp_path / 'origin' / 'case_002' / 'pr3'
    origin.mkdir(parents=True)
    ref = lambda name: {'file': str(origin / name), 'json_path': []}
    unit = {'candidate_id': 'old0', 'search_position': {'layer': 1, 'offset': 1}}
    constructor = {field: ref(field) for field in (
        'pattern_ref', 'holes_ref', 'components_ref', 'program_ref')}
    source_reg = {'material': {'source_training_ref': ref('training.json'),
        'common_executable_model_roster_ref': ref('models.json'),
        'parent_constructor_source': constructor},
        'arms': {'remove': {'candidate_versions': [{'id': 'old0', 'version': 1}]}},
        'parent_research': {'complete_unit_compute_limits': {'remove': [unit]}}}
    (source.parent / 'registration.json').write_text(json.dumps(source_reg))
    case = {'id': 'case_002', 'kind': 'delay', 'seed': 17, 'confirmation_seed': 29}
    plan = {'cases': [case], 'prepared_public_questions': {'case_002': {'id': 'readout'}}}
    (origin.parent.parent / 'protocol.json').write_text(json.dumps(plan))
    (source / 'registered_actions.json').write_text(json.dumps([{
        'action': 'research_choice', 'complete_model_roster_ref': ref('inherited_bad.json')}]))
    (source / 'final_costs.json').write_text(json.dumps(costs(999)))
    previous = {'costs': costs(10), 'online_questions': [{'id': 'stale'}],
        'theory_index': {'old': {}}, 'paid_observations': ['retained97'],
        'research_organization': {'old': True}}
    paid_ref = {'question_id': 'native_public_response', 'meaning_version': 1}
    readout = {'id': paid_ref['question_id'], 'meaning_version': 1,
        'kind': 'value', 'query_semantics': {'mode': 'model_query'},
        'context_ref': {'name': 'same_paid_rows'}}
    duty = {'id': 'parent195', 'meaning_version': 1, 'kind': 'program_revision',
        'context_ref': readout['context_ref'], 'paid_score_question_ref': paid_ref,
        'native_research_ref': deepcopy(paid_ref),
        'theory_target': {'parent_theory_id': 'actual_parent195'}}
    scope_ref = ref('scope.json')
    state = {'costs': costs(11), 'questions': [readout, duty],
        'revisions': [], 'theory_index': {'old': {}, 'paid7': {}},
        'research_organization': {'actual195': True}, 'operator_results': [
            {'index': 0, 'operator': 'research_choice', 'truncated': False,
             'models_requested': 2, 'models_completed': 2, 'paid_rows': 97,
             'evaluations': [{'path': model, 'paid_score_reuse': {'scope_ref': scope_ref}}
                for model in ['strong.py', 'paid7.py']],
             'arguments': {'operator': 'research_choice',
                'models': ['strong.py', 'paid7.py']}}]}
    progress = {'source_workspace': str(source), 'state_costs': costs(11),
        'fully_committed_new_units': 0, 'native_units_executed': 1,
        'comparison_returned_but_complete_checkpoint_uncommitted_units': 1,
        'unit_receipts': [{'remaining_plan_index': 0, 'candidate_id': 'old0',
            'registered_position': unit['search_position'],
            'actual_persisted_position': unit['search_position'],
            'native_unit_completed': True, 'comparison_returned_and_checkpoint_started': True,
            'complete_checkpoint_committed': False}]}
    registration = {'material': {'same_closed_scene': str(source),
        'training_ref': {'file': str(source / 'training.json'), 'json_path': []},
        'new_object_ref': ref('new.json'), 'paid_rows': 97},
        'parent_research': {'question_id': 'parent195', 'meaning_version': 1,
            'source_question_id': 'native_public_response',
            'parent_theory_ref': {'id': 'actual_parent195', 'version': 1}}}
    values = {'training.json': previous,
        'initial_archive.json': {'files': {}, 'finite_native_author': {}},
        'models.json': ['strong.py'], 'new.json': {'id': 'new20'},
        'scope.json': {'question': deepcopy(readout), 'rows': [None] * 97}}
    reads = []

    def read(reference, workspace):
        name = Path(reference['file']).name
        reads.append(name)
        return deepcopy(values[name])

    monkeypatch.setattr(scenes, '_read', read)
    monkeypatch.setattr(scenes, 'resolve_workspace_archive', lambda value: value)
    monkeypatch.setattr(scenes, 'load_source_values',
        lambda references, workspace: [{'value': {}} for item in references])
    return registration, state, progress, previous, reads


def test_interrupted_loader_uses_CP8_roster_and_costs_not_inherited_terminal_files(tmp_path, monkeypatch):
    registration, state, progress, previous, reads = interrupted_material(tmp_path, monkeypatch)
    material = scenes.load_completed_source_material(registration,
        retained_state=state, interrupted_progress=progress)
    assert material['previous']['online_questions'] is state['questions']
    assert material['previous']['theory_index'] is state['theory_index']
    assert material['previous']['research_organization'] is state['research_organization']
    assert material['previous']['paid_observations'] == previous['paid_observations']
    assert material['models'] == ['strong.py', 'paid7.py']
    assert material['starting_costs'] == costs(11)
    assert material['retained_training_costs'] == costs(10)
    assert material['closed_scene_theory_count'] == 2
    assert material['retained_state'] is state
    assert 'inherited_bad.json' not in reads
    assert reads.count('scope.json') == 1
    assert material['retained_score_binding']['paid_score_question_ref'] == {
        'question_id': 'native_public_response', 'meaning_version': 1}
    assert material['retained_comparison_roster_ref']['json_path'] == [
        'operator_results', 0, 'arguments', 'models']


def test_comparison_cannot_reuse_an_unrelated_question_scope(tmp_path, monkeypatch):
    registration, state, progress, _, _ = interrupted_material(tmp_path, monkeypatch)
    original_read = scenes._read

    def unrelated_scope(reference, workspace):
        value = original_read(reference, workspace)
        if Path(reference['file']).name == 'scope.json':
            value['question']['id'] = 'unrelated_readout'
        return value

    monkeypatch.setattr(scenes, '_read', unrelated_scope)
    with pytest.raises(ValueError, match='scope differs from its source task'):
        scenes.load_completed_source_material(registration,
            retained_state=state, interrupted_progress=progress)


@pytest.mark.parametrize('change', ['owner', 'position', 'cost', 'source', 'truncated',
    'unknown_task', 'different_meaning', 'missing_scope', 'missing_model'])
def test_interrupted_loader_rejects_unverified_prefix_or_incomplete_comparison(tmp_path, monkeypatch, change):
    registration, state, progress, _, _ = interrupted_material(tmp_path, monkeypatch)
    if change == 'owner':
        progress['unit_receipts'][0]['candidate_id'] = 'new20'
    elif change == 'position':
        progress['unit_receipts'][0]['actual_persisted_position'] = {'layer': 0, 'offset': 1}
    elif change == 'cost':
        progress['state_costs']['compute_cost'] = 12
    elif change == 'source':
        progress['source_workspace'] = str(tmp_path / 'other')
    elif change == 'truncated':
        state['operator_results'][0]['truncated'] = True
    elif change == 'unknown_task':
        state['questions'][1]['paid_score_question_ref'] = {
            'question_id': 'unrelated_question', 'meaning_version': 1}
    elif change == 'different_meaning':
        state['questions'][0]['meaning_version'] = 2
    elif change == 'missing_scope':
        del state['operator_results'][0]['evaluations'][0]['paid_score_reuse']
    else:
        state['operator_results'][0]['evaluations'].pop()
    with pytest.raises(ValueError):
        scenes.load_completed_source_material(registration,
            retained_state=state, interrupted_progress=progress)


def test_current_actions_are_saved_before_the_large_checkpoint_can_interrupt(tmp_path, monkeypatch):
    from test_registered_parent_revision_control import runner_scene
    scene, registration, calls, _, _ = runner_scene(tmp_path, monkeypatch)
    snapshots = []

    def interrupt(agent, status):
        actions = json.loads((tmp_path / 'registered_actions.json').read_text())
        snapshots.append(actions)
        if len(snapshots) == 2:
            raise KeyboardInterrupt('global OOM during the large checkpoint')

    monkeypatch.setattr(control, '_checkpoint', interrupt)
    with pytest.raises(KeyboardInterrupt):
        control._registered_parent_revision_arm_run(scene, registration)
    paid = snapshots[1]
    assert [row['action'] for row in paid] == ['research_choice',
        'frozen_recall_on_registered_parent_duty', 'research_choice']
    assert paid[1]['feedback']['complete_research_visits'][0]['candidate_id'] == 'old'
    assert paid[1]['feedback']['complete_research_visits'][0]['search_position'] == {
        'layer': 0, 'offset': 1}
    assert paid[2]['models_requested'] == 2847
    assert calls == ['compare', 'recall', 'compare']
