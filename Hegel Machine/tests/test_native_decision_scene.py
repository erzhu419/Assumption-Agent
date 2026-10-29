"""Restore original decision inputs, never the later successful candidate pool."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, TaskSpec
from scripts.native_decision_scene import recover_native_decision_scene


def fixture_scene():
    parent = Program('feature', name='x').to_dict()
    model = {'model_id': 'objects_1', 'groups': [{'object_id': 'a', 'q': 'x'}],
        'support_rows': 3, 'acquisition_rows': 3,
        'schema': {'objects': ['a'], 'fields': {'a': {'q': 'x'}}}}
    rows = lambda values, feature='x': [{'features': {feature: v}, 'target': v,
        'history': [{feature: v - 1}]} for v in values]
    report = {'task_id': 'anonymous_process', 'actions': [
        {'action': 'acquire', 'probe_id': 'fit'},
        {'action': 'adopt_discovered_objects', 'groups': model['groups'], 'object_model_id': 'objects_1'},
        {'action': 'acquire', 'probe_id': 'commands'},
        {'action': 'acquire', 'probe_id': 'compare'},
        {'action': 'compare_partial_object_claims', 'alternatives': []},
        {'action': 'choose_research_action', 'selected_action': 'program_replay',
            'eligible_actions': ['program_replay', 'structural_research'],
            'fallback_action': 'structural_research', 'context': 'structured:parent',
            'compute_remaining': 82868, 'observation_remaining': 389, 'measurement_remaining': 5181},
        {'action': 'research_action_feedback', 'research_action': 'program_replay',
            'evidence_basis': 'common_unfitted_validation', 'fitting_rows': 3, 'evidence_rows': 2},
        {'action': 'propose_mechanism_revisions', 'alternatives': [
            {'revision': {'original_program': parent}}]},
        {'action': 'execute_dependent_research', 'parent_program': parent,
            'parent_candidate_id': 'old_parent'},
        {'action': 'compare_research_directions', 'alternatives': [
            {'kind': 'current_affine', 'validation_loss': .5},
            {'kind': 'retained_response', 'validation_loss': .25}]},
        {'action': 'acquire', 'probe_id': 'later'}],
        'batches': [
            {'probe': {'probe_id': 'fit'}, 'observations': rows((1, 2, 3)),
                'transitions': [{'before': {'x': 1}, 'after': {'x': 2}}]},
            {'probe': {'probe_id': 'commands'}, 'observations': rows((4, 5), 'command')},
            {'probe': {'probe_id': 'compare'}, 'observations': rows((6, 7))},
            {'probe': {'probe_id': 'later'}, 'observations': rows((999,))}],
        'object_continuation_state': {'candidate_frontiers': [{'program': 'future'}]},
        'object_research': {'model_id': 'future_roles'}}
    registration = {'historical_knowledge_ref': {'store': 'retained.sqlite', 'sequence': 10},
                    'research_config': {'max_history': 4}}
    arguments = {'public_spec': TaskSpec('anonymous_process', 'public', ('x',), (), ()),
        'report_ref': {'file': 'response_report.json'},
        'registration_ref': {'file': 'registration.json'}, 'model_ref': {'file': 'parent_result.json'}}
    return report, registration, model, arguments


def test_restore_first_choice_preserves_full_trajectories_without_future_pool_or_science(monkeypatch):
    def no_prediction(*args, **kwargs):
        raise AssertionError('Input recovery must not predict or fit')
    monkeypatch.setattr(Program, 'predict', no_prediction)
    report, registration, model, arguments = fixture_scene()
    original = deepcopy(report)
    scene = recover_native_decision_scene(report, registration, model, **arguments)
    assert report == original
    assert [r.target for r in scene['training']] == [1, 2, 3]
    assert [r.target for r in scene['validation']] == [6, 7]
    assert scene['training'][1].history == ({'x': 1},)
    assert scene['transitions'] == [{'before': {'x': 1}, 'after': {'x': 2}}]
    assert scene['choice_index'] == 5 and scene['ordinary_operation'] == 'program_replay'
    assert scene['fallback_operation'] == 'structural_research'
    assert scene['remaining_compute'] == 82868
    assert scene['parent']['candidate'].program == Program('feature', name='x')
    assert scene['references']['parent_original']['json_path'] == [
        'actions', 7, 'alternatives', 0, 'revision', 'original_program']
    assert scene['references']['validation']['json_path'] == ['batches', 2]
    assert scene['historical_knowledge_ref'] == registration['historical_knowledge_ref']
    assert scene['complete_historical_pool_available'] is False
    assert scene['preparation_cost_scope'] == 'new_shared_preparation_not_free_historical_cache'
    assert 'pool' not in scene and 'candidate_frontiers' not in scene
    assert scene['spec'].context['object_model']['model_id'] == 'objects_1'


def test_future_terminal_changes_cannot_change_restored_inputs():
    report, registration, model, arguments = fixture_scene()
    first = recover_native_decision_scene(report, registration, model, **arguments)
    report['object_continuation_state'] = {'selected': Program('const', value=-999).to_dict()}
    report['object_research'] = {'schema': 'future_model'}
    report['batches'][-1]['observations'][0]['target'] = -999
    second = recover_native_decision_scene(report, registration, model, **arguments)
    assert first == second


def test_missing_original_parent_does_not_fall_back_to_terminal_predictor():
    report, registration, model, arguments = fixture_scene()
    report['actions'][7]['alternatives'] = []
    with pytest.raises(ValueError, match='unchanged original-program carrier'):
        recover_native_decision_scene(report, registration, model, **arguments)


def test_original_command_binding_recovers_a_choice_without_dependent_execution(monkeypatch):
    report, registration, model, arguments = fixture_scene()
    del report['actions'][8]  # The original parent can remain unused by this direction.
    bound = {'candidate': Candidate('retained:object_recovery',
        Program('feature', name='x'), 'retained_response')}
    def no_prediction(*args, **kwargs):
        raise AssertionError('Restoring the old binding must not fit or predict')
    monkeypatch.setattr(Program, 'predict', no_prediction)
    scene = recover_native_decision_scene(report, registration, model,
        reconstructed_parent=bound, **arguments)
    assert scene['parent']['candidate'].candidate_id == 'retained:object_recovery'
    assert scene['parent']['candidate'].program == bound['candidate'].program
    assert scene['parent']['validation_loss'] == .25
    assert scene['references']['parent_carrier']['json_path'] == [
        'actions', 7, 'alternatives', 0, 'revision', 'original_program']
    assert scene['expected_preparation']['retained_response_index'] == 1
    assert not scene['complete_historical_pool_available']


@pytest.mark.parametrize('bound', [None, {'candidate': Candidate('later',
    Program('const', value=999), 'retained_response')}])
def test_missing_or_mismatched_command_binding_cannot_supply_a_parent(bound):
    report, registration, model, arguments = fixture_scene()
    del report['actions'][8]
    with pytest.raises(ValueError, match='original command parent|unchanged original-program carrier'):
        recover_native_decision_scene(report, registration, model,
            reconstructed_parent=bound, **arguments)


def test_wrong_paid_cutoff_or_role_state_stops_recovery():
    report, registration, model, arguments = fixture_scene()
    report['actions'][6]['fitting_rows'] = 4
    with pytest.raises(ValueError, match='trajectory cutoffs'):
        recover_native_decision_scene(report, registration, model, **arguments)
    report['actions'][6]['fitting_rows'] = 3
    model['support_rows'] = 4
    with pytest.raises(ValueError, match='role model'):
        recover_native_decision_scene(report, registration, model, **arguments)


def second_stage_scene():
    report, registration, model, arguments = fixture_scene()
    parent = Program('feature', name='x').to_dict()
    kind = 'partial_object:window_1_totals'
    graph = {'graph': {'nodes': {'answer': {'answer': parent, 'loss': .11}}},
             'output_node': 'answer', 'fit_rows': 3, 'term_nodes': []}
    report['actions'][9]['selected_kind'] = kind
    report['actions'].extend([
        {'action': 'validate_frozen_program', 'program': parent, 'loss': .7, 'accepted': False},
        {'action': 'object_research_feedback', 'program': parent, 'status': 'refuted'},
        {'action': 'acquire', 'probe_id': 'second_compare'},
        {'action': 'compare_partial_object_claims', 'alternatives': [
            {'kind': kind, 'validation_loss': .2}]},
        {'action': 'choose_research_action', 'selected_action': 'program_revision',
            'eligible_actions': ['program_revision', 'program_replay', 'structural_research'],
            'context': 'structured:parent', 'compute_remaining': 78932,
            'observation_remaining': 341, 'measurement_remaining': 4749},
        {'action': 'research_action_feedback', 'research_action': 'program_revision',
            'evidence_basis': 'common_unfitted_validation', 'fitting_rows': 6, 'evidence_rows': 2},
        {'action': 'propose_mechanism_revisions', 'alternatives': [
            {'revision': {'original_program': parent, 'original_research_program': graph}}]},
        {'action': 'compare_research_directions', 'alternatives': [
            {'kind': 'current_affine', 'validation_loss': .8},
            {'kind': kind, 'validation_loss': .2},
            {'kind': kind, 'validation_loss': .3}]}])
    report['batches'].append({'probe': {'probe_id': 'second_compare'},
        'observations': [{'features': {'x': v}, 'target': v, 'history': []} for v in (8, 9)]})
    report['object_continuation_state']['candidate_frontiers'] = [{'entries': [{
        'kind': kind, 'candidate': {'program': Program('const', value=-999).to_dict()},
        'validation_loss': -999, 'research_program': {'future': True},
        'partial_object_knowledge': {'source_model_id': model['model_id'], 'used_fields': ['x'],
            'role_coverage': {'q': ['a']}, 'unresolved': []}}]}]
    return report, registration, model, arguments


def test_restore_new_data_stage_keeps_old_model_parent_graph_and_distinct_same_kind_scores(monkeypatch):
    def no_science(*args, **kwargs):
        raise AssertionError('A new data cutoff is restored without science')
    monkeypatch.setattr(Program, 'predict', no_science)
    report, registration, model, arguments = second_stage_scene()
    scene = recover_native_decision_scene(report, registration, model, decision_index=15, **arguments)
    assert [row.target for row in scene['training']] == [1, 2, 3, 6, 7, 999]
    assert [row.target for row in scene['validation']] == [8, 9]
    assert scene['object_model']['support_rows'] == 3
    assert scene['ordinary_operation'] == 'program_revision'
    assert scene['fallback_operation'] is None
    assert scene['parent']['research_program']['fit_rows'] == 3
    assert scene['parent']['validation_loss'] == .3
    assert scene['expected_preparation']['partial_claims'][0]['validation_loss'] == .2
    assert scene['expected_preparation']['retained_response_index'] == 2
    assert scene['references']['parent_carrier']['json_path'] == ['actions', 11, 'program']
    assert scene['parent']['candidate'].program == Program('feature', name='x')
    assert scene['parent']['partial_object_knowledge']['used_fields'] == ['x']


def test_middle_of_stage_is_rejected_when_earlier_new_candidates_have_no_complete_checkpoint():
    report, registration, model, arguments = second_stage_scene()
    report['actions'].insert(17, {**report['actions'][15], 'selected_action': 'program_replay'})
    with pytest.raises(ValueError, match='stage first choice'):
        recover_native_decision_scene(report, registration, model, decision_index=17, **arguments)


@pytest.mark.parametrize('decision_index', [None, 15])
def test_later_knowledge_keeps_original_parent_and_separate_target_identity(
        tmp_path, monkeypatch, decision_index):
    from scripts import native_decision_scene as loader
    from scripts.run_partial_role_parent import save
    from hegel_machine.cognition.memory import KnowledgeLibrary

    report, registration, model, arguments = (fixture_scene() if decision_index is None
                                               else second_stage_scene())
    for action in report['actions']:
        if action['action'] == 'object_research_feedback':
            action.update(direction='program_replay', context='old_context', feedback={'uses': 1})
    registration['tasks'] = [{'seed': 123}]
    source = tmp_path / 'history'
    scene_path = source / 'runs/123/pr3'
    save(source / 'registration.json', registration)
    save(scene_path / 'response_report.json', report)
    save(scene_path / 'common/object_question_000/parent_result.json', model)
    historical = KnowledgeLibrary().to_dict()
    frozen = KnowledgeLibrary(object_models=[{'model_id': model['model_id'], 'source': 'later_task'}],
        program_experiment_methods=[{'method_id': 'later_method', 'status': 'supported'}])
    frozen.action_experience = {'frozen_context': {'program_replay': {'uses': 3}}}
    frozen_before = deepcopy(frozen.to_dict())
    monkeypatch.setattr('scripts.run_partial_role_successors.historical_knowledge',
                        lambda location: (deepcopy(historical), {'compute_cost': 100}, 10))
    monkeypatch.setattr('scripts.successor_interval_scene.load_confirmed_knowledge', lambda path: frozen)

    class PublicOnlyLab:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def spec_for(self, channels):
            return arguments['public_spec']

    monkeypatch.setattr(loader, 'AnonymousObjectLab', PublicOnlyLab)
    monkeypatch.setattr(Program, 'predict', lambda *a, **kw: pytest.fail('Restore must not predict'))
    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit',
                        lambda *a, **kw: pytest.fail('Restore must not fit'))
    old = loader.load_native_decision_scene(source, seed=123, decision_index=decision_index)
    files_before = {path: path.read_bytes() for path in source.rglob('*.json')}
    current = loader.load_native_decision_scene(source, seed=123, decision_index=decision_index,
                                               knowledge_root=tmp_path / 'K2')

    assert current['parent']['candidate'] == old['parent']['candidate']
    assert current['parent']['validation_loss'] == old['parent']['validation_loss']
    assert current['training'] == old['training'] and current['validation'] == old['validation']
    assert current['expected_preparation'] == old['expected_preparation']
    assert current['references'] == old['references']
    assert current['knowledge']['object_models'][0] == frozen_before['object_models'][0]
    assert current['object_model']['model_id'] == 'objects_0002'
    assert current['knowledge']['object_models'][1] == current['object_model']
    assert current['spec'].context['object_model']['model_id'] == 'objects_0002'
    assert current['spec'].context['object_schema'] == old['spec'].context['object_schema']
    assert current['knowledge']['action_experience'] == frozen_before['action_experience']
    assert not current['action_experience_restored_from']
    assert current['historical_action_experience_refs'] == old['action_experience_restored_from']
    assert current['knowledge']['program_experiment_methods'] == frozen_before['program_experiment_methods']
    assert current['knowledge_base_ref'] == {'file': str(tmp_path / 'K2/knowledge_delta.json')}
    assert current['knowledge_source_scope'] == 'retrospective_development_with_frozen_later_knowledge'
    assert current['object_identity_remap'] == {'original_model_id': model['model_id'],
        'current_model_id': 'objects_0002', 'source_ref': old['references']['object_model'],
        'reason': 'retain_target_object_in_frozen_later_library'}
    if decision_index is not None:
        assert current['parent']['partial_object_knowledge']['source_model_id'] == 'objects_0002'
        assert old['parent']['partial_object_knowledge']['source_model_id'] == model['model_id']
    assert frozen.to_dict() == frozen_before
    assert {path: path.read_bytes() for path in source.rglob('*.json')} == files_before
