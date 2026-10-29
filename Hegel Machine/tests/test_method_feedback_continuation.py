from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint, write_json_checkpoint
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.models import Observation, TaskSpec
from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.language import Program


from scripts import run_method_feedback_continuation as driver


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_checkpoint(path, value, indent=None)
    return driver.ref(path)


def test_feedback_host_window_escapes_nested_python_catch_and_retains_the_prior_account(
        tmp_path, monkeypatch):
    from time import sleep
    from hegel_machine.cognition.experiment_runtime import PythonExecutionDeadline, python_execution_deadline
    prior = save(tmp_path / 'old_stage.json', {'costs': {'compute_cost': 23,
        'python_execution_seconds': 0}, 'native_host_wall_seconds': 0.})
    monkeypatch.setattr(driver, 'NATIVE_HOST_CAP', .05)

    def delayed_restore(*args, **kwargs):
        try:
            with python_execution_deadline(.5):
                sleep(.2)
        except PythonExecutionDeadline:
            pytest.fail('the whole feedback host cutoff became a local Python return')

    monkeypatch.setattr(driver, 'CommonArchiveInquiryAgent', delayed_restore)
    manifest = {'selected_actual_use_refs': [{'file': 'retained_paid_use.json', 'json_path': []}]}
    result = driver.run_feedback(manifest, tmp_path / 'interrupted_feedback', stage1=prior['file'])
    assert result['status'] == 'time_exhausted'
    assert result['error']['type'] == 'ActivityExecutionDeadline'
    assert result['cumulative_stage1_costs']['compute_cost'] == 23
    assert result['new_costs']['compute_cost'] == 0
    assert result['new_world_fits'] == result['new_world_predictions'] == result['new_world_observations'] == 0


ROLES = {'role_0': {'role': 'public_control', 'unit': '1', 'time': 'current'}}
FIT_CONTRACT = {'operation': 'retained_fixture_only'}


def native(workspace, number, *, origin=None, complete=False, helpful=False, constant=False):
    rows = [{'features': {'u': -1. + 2*i/7}, 'target': 0. if constant else float(i),
        'history': [{'u': -1. + 2*j/7} for j in range(i-i%4, i)]} for i in range(8)]
    observations = [{'action': {'kind': 'trajectory', 'sequence': [row['features'] for row in rows[start:start+4]]},
        'rows': rows[start:start+4], 'row_indices': list(range(start,start+4))} for start in (0,4)]
    save(workspace/'laboratory_observations.json', observations)
    paid = save(workspace / 'method_feedback/paid.json', rows)
    profile = method_profile(TaskSpec('fixture', 'Paid public response', ('u',), (), (),
        feature_bounds={'u': (-1.,1.)}, max_probe_length=24),
        [Observation(row['features'], row['target'], tuple(row['history'])) for row in rows],
        minimum_improvement=1e-10)
    profile['representation'].pop('program')
    profile_ref = save(workspace / 'method_feedback/profile.json', profile)
    binding = {'kind': 'research_program', 'mapping': {'role_0': 'u'},
        'role_signature': {'u': deepcopy(ROLES['role_0'])}, 'output_unit': '1',
        'context_ref': {'source_id': str(workspace)},
        'fit_split': {'records': [0,1], 'training_records': [0], 'validation_records': [1],
            'requested_validation_records': [1], 'duplicate_groups': [], 'train_rows': 4, 'holdout_rows': 4},
        'paid_cutoff': {'records': 2, 'rows': 8}, 'source_refs': [],
        'source_seed_inputs': [{'data': 'old'}], 'seed_policy': 'retained_fixture',
        'research_program_contract': deepcopy(FIT_CONTRACT)}
    record = {'method_ref': {'id': driver.METHOD, 'version': 2},
        'invoked_method_ref': {'id': driver.METHOD, 'version': 5},
        'status': 'completed' if complete else 'execution_error', 'execution_complete': complete,
        'fit_compute_cost': 80, 'cost_refs': [driver.ref(workspace / 'paid_fee.json')],
        'profile_ref': profile_ref,
        'evidence_basis': {**deepcopy(binding), 'paid_evidence_ref': paid},
        'invocation_binding': binding,
        'parent_question_ref': {'question_id': 'native_public_response', 'meaning_version': 1}}
    if complete:
        record['working_comparison'] = {'complete': True, 'same_rows': True, 'rows': 8,
            'data_role': 'exposed_development', 'before_loss': 1., 'after_loss': .5 if helpful else 2.}
    else:
        record['execution_error'] = {'type': 'KeyError', 'message': 'retained_checks'}
    if origin:
        record['origin_ref'] = origin
    reference = save(workspace / f'method_feedback/use_{number:06d}.json', record)
    return reference


def publish_calls(workspace, calls):
    save(workspace / 'shared/theory_library.json', {'method_feedback': {
        driver.METHOD: {'outcome_refs': calls}}})


def cohort(tmp_path):
    original, final = tmp_path / 'original', tmp_path / 'recovery/target'
    cases = [{'id': f'case_{i:03d}', 'kind': kind, 'seed': 100+i}
        for i, kind in enumerate(('memoryless', 'delay', 'hysteresis'))]
    arms = ['archive', 'fixed_pr2', 'pr3']
    trials, sidecar, completed = [], [], []
    for case in cases:
        for arm in arms:
            old = original / 'target' / case['id'] / arm
            new = final / case['id'] / arm
            result_ref = driver.ref(new / 'training.json')
            trials.append({'case': case['id'], 'arm': arm, 'status': 'no_submitted_model',
                'terminal': {'status': 'no_submitted_model', 'evaluation_rows': 0}})
            sidecar.append({'case': case, 'arm': arm, 'result_ref': result_ref})
            completed.append({'case': case['id'], 'arm': arm, 'checkpoint_ref': driver.ref(old/'training.json')})
    save(final/'summary.json', {'trials': trials})
    save(final/'training_results.json', sidecar)
    save(final/'protocol.json', {'cases': cases, 'arms': arms})
    save(original/'interruption_recovery/stopped_thin_admission.json', {
        'completed_training_reuse': completed, 'tasks': []})
    return original, final


def test_first_actual_c_is_retained_even_if_later_b_or_c_is_better(tmp_path, monkeypatch):
    original, final = cohort(tmp_path)
    old_c = original/'target/case_000/pr3'
    new_c = final/'case_000/pr3'
    first = native(old_c, 0, complete=False)
    later = native(old_c, 1, complete=True, helpful=True)
    updated = native(new_c, 2, origin=first, complete=False)
    publish_calls(old_c, [first, later])
    publish_calls(new_c, [first, later, updated])
    old_b = original/'target/case_000/fixed_pr2'
    b = native(old_b, 0, complete=True, helpful=True)
    publish_calls(old_b, [b])
    case1_b = final/'case_001/fixed_pr2'
    b1 = native(case1_b, 0, complete=True)
    # The inherited use cannot give case_001 a fictitious C invocation.
    publish_calls(final/'case_001/pr3', [first])
    publish_calls(case1_b, [b1])
    def forbidden(*args, **kwargs):
        raise AssertionError('manifest construction must not decode a whole training result')
    original_reader = driver.read_json_checkpoint
    monkeypatch.setattr(driver, 'read_json_checkpoint', lambda path, **kw:
        forbidden() if Path(path).name == 'training.json' else original_reader(path, **kw))
    manifest = driver.prepare_manifest(final/'summary.json', original)
    assert manifest['selected_actual_use_refs'] == [updated, b1]
    c0, c1, c2 = manifest['physical_cases']
    assert c0['selected_arm'] == 'pr3'
    assert c0['selected_actual_use']['origin_ref'] == first
    assert c0['selected_actual_use']['status'] == 'execution_error'
    assert len(c0['all_actual_uses']['pr3']) == 2
    assert c1['selected_arm'] == 'fixed_pr2'
    assert c2['selected_actual_use'] is None
    assert all(c['body_process_return_from_policy_contrast'] == 'UNKNOWN' for c in manifest['physical_cases'])


def baseline(tmp_path):
    outcomes = save(tmp_path/'baseline/empty_conditions.json', [])
    method = {'id': driver.METHOD, 'kind': 'research_program', 'version': 5,
        'method_use_conditions': {'usage_version': {'id': driver.METHOD, 'version': 2}, 'outcomes_ref': outcomes},
        'research_program_contract': deepcopy(FIT_CONTRACT), 'source_seed_inputs': [{'data': 'old'}],
        'template': AbstractTemplate('method_fixture', 'paid_fixture', Program('feature', name='role_0'),
            ('role_0',), ('u',), ()).to_dict(), 'role_signature': deepcopy(ROLES), 'output_unit': '1',
        'scope': {'context_ref': 'fixture_public'}, 'research_program_source_refs': [],
        'seed_policy': 'retained_fixture'}
    candidates = {f'old_{i}': {'id': f'old_{i}', 'version': 1, 'kind': 'retained_fixture'} for i in range(20)}
    candidates[driver.METHOD] = method
    reference = save(tmp_path/'baseline/library.json', {'candidates': candidates})
    return reference, candidates


def test_actual_common_import_route_preserves_body_fees_and_censored_unknown_without_science(tmp_path, monkeypatch):
    original, final = cohort(tmp_path)
    workspace = final/'case_000/pr3'
    actual = native(workspace, 0, complete=True, helpful=True)
    later = native(workspace, 1, complete=False)
    publish_calls(workspace, [actual, later])
    manifest = driver.prepare_manifest(final/'summary.json', original)
    old_ref, candidates = baseline(tmp_path)
    prior = save(tmp_path/'stage1.json', {'costs': {'compute_cost': 283200,
        'python_execution_seconds': 0}, 'native_host_wall_seconds': 24.034804177004844})
    def forbidden(*args, **kwargs):
        raise AssertionError('native feedback must never fit, score a world model, buy data or call a provider')
    import hegel_machine.cognition.research_programs as programs
    import hegel_machine.cognition.theory_consolidation as theory
    import hegel_machine.cognition.code_agent as code
    monkeypatch.setattr(programs, 'replay_research_program', forbidden)
    monkeypatch.setattr(theory, '_transfer_research_program', forbidden)
    monkeypatch.setattr(code, 'evaluate_python_model', forbidden)
    monkeypatch.setattr(driver.CommonArchiveInquiryAgent, '_request', forbidden)
    result = driver.run_feedback(manifest, tmp_path/'feedback_run', baseline=old_ref['file'], stage1=prior['file'])
    assert result['status'] == 'completed_feedback_only', result.get('error')
    assert result['library_candidate_count'] == 21
    assert result['new_condition_version'] > 5
    assert result['new_costs']['compute_cost'] > 0
    assert result['cumulative_stage1_costs']['compute_cost'] == 283200 + result['new_costs']['compute_cost']
    assert 24.034804177004844 < result['cumulative_stage1_costs']['native_host_wall_seconds'] < 600
    assert not result['next_scientific_action_executed']
    assert all(result[k] == 0 for k in ('new_world_fits','new_world_predictions','new_world_observations','new_terminal_evaluations'))
    assert result['actual_agent_costs']['observation_cost'] == result['actual_agent_costs']['model_calls'] == 0
    library = read_json_checkpoint(result['learned_library_ref']['file'])
    group = library['method_feedback'][driver.METHOD]
    assert len(group['outcome_refs']) == 1
    imported = driver.read(group['outcome_refs'][0], tmp_path)
    assert imported['working_comparison']['after_loss'] == .5
    assert imported['fit_compute_cost'] == 80
    assert 'paired_future' not in imported and 'process_comparison_ref' not in imported
    assert group['returns']['origins'][0]['status'] == 'unscored'
    assert group['returns']['origins'][0]['fit_compute_cost'] == 80
    assert result['ordinary_decisions'][0]['scope'].startswith('same_paid_profile')
    # Whole library remains by reference; local changed fields never replace it.
    assert all(identifier in candidates for identifier in library.get('candidates', {}))


def test_partial_final_summary_waits_without_inventing_a_case(tmp_path):
    original, final = cohort(tmp_path)
    summary = read_json_checkpoint(final/'summary.json')
    summary['trials'].pop()
    save(final/'summary.json', summary)
    with pytest.raises(ValueError, match='complete final cohort'):
        driver.prepare_manifest(final/'summary.json', original)


def test_exhausted_old_compute_is_visible_and_never_renews_the_window(tmp_path, monkeypatch):
    original, final = cohort(tmp_path)
    workspace = final/'case_000/pr3'
    actual = native(workspace, 0, complete=True, helpful=True)
    publish_calls(workspace, [actual])
    manifest = driver.prepare_manifest(final/'summary.json', original)
    old_ref, _ = baseline(tmp_path)
    prior = save(tmp_path/'stage1.json', {'costs': {'compute_cost': 283200,
        'python_execution_seconds': 0}, 'native_host_wall_seconds': 24.034804177004844})
    monkeypatch.setattr(driver, 'COMPUTE_CAP', 283201)
    result = driver.run_feedback(manifest, tmp_path/'feedback_run', baseline=old_ref['file'], stage1=prior['file'])
    assert result['status'] == 'feedback_recorded_condition_research_incomplete', result.get('error')
    assert result['new_costs']['compute_cost'] == 1
    assert result['cumulative_stage1_costs']['compute_cost'] == result['cumulative_stage1_costs']['compute_cap']
    assert result['new_condition_version'] == 5
    assert result['research_route']['status'] == 'scoped_unknown'
    assert result['research_route']['next_action'] is None


def test_two_actual_profiles_route_first_obligation_and_execute_only_real_common_defer(tmp_path, monkeypatch):
    original, final = cohort(tmp_path)
    references = []
    for index, constant in ((0,True),(1,False)):
        workspace = final/f'case_{index:03d}/pr3'
        reference = native(workspace, 0, complete=True, helpful=not constant, constant=constant)
        publish_calls(workspace, [reference])
        references.append(reference)
    manifest = driver.prepare_manifest(final/'summary.json', original)
    old_ref, _ = baseline(tmp_path)
    prior = save(tmp_path/'stage1.json', {'costs': {'compute_cost': 283200,
        'python_execution_seconds': 0}, 'native_host_wall_seconds': 24.034804177004844})
    actual_execute = driver.CommonArchiveInquiryAgent._execute
    observed = []
    def traced(self, action, arguments, laboratory):
        if arguments.get('operator') == 'research_route':
            observed.append([row.target for row in self.rows])
            assert len(self.observations) == 2
            assert self._method_use_profile['data']['response_levels'] == 1
        return actual_execute(self, action, arguments, laboratory)
    monkeypatch.setattr(driver.CommonArchiveInquiryAgent, '_execute', traced)
    def forbidden(*args, **kwargs):
        raise AssertionError('the actual defer action must not fit, predict or buy observations')
    import hegel_machine.cognition.research_programs as programs
    import hegel_machine.cognition.code_agent as code
    monkeypatch.setattr(programs, 'replay_research_program', forbidden)
    monkeypatch.setattr(code, 'evaluate_python_model', forbidden)
    monkeypatch.setattr(driver.AnonymousHistoryLab, 'execute', forbidden)
    monkeypatch.setattr(driver.AnonymousHistoryLab, 'prediction_rows', forbidden)
    monkeypatch.setattr(driver.AnonymousHistoryLab, 'evaluation_rows', forbidden)
    result = driver.run_feedback(manifest, tmp_path/'feedback_run', baseline=old_ref['file'], stage1=prior['file'])
    assert result['status'] == 'completed_feedback_only', result.get('error')
    assert observed == [[0.]*8]
    assert result['condition_route_same_pre_call_profile'] is True
    first, second = result['ordinary_decisions']
    assert first['new_ordinary_admission']['admission'] == 'defer'
    assert first['ordinary_defer_same_pre_call_profile'] is True
    assert first['ordinary_defer_executed'] is True
    assert first['ordinary_common_action']['status'] == 'method_use_deferred'
    assert first['ordinary_common_action']['new_compute_cost'] == 0
    assert second['new_ordinary_admission']['admission'] == 'admit'
    assert 'ordinary_common_action' not in second
    assert result['actual_agent_costs']['observation_cost'] == result['actual_agent_costs']['model_calls'] == 0


def paid_parent_scene(tmp_path, *, retained=True):
    workspace = tmp_path / 'original'
    reference = native(workspace, 0, complete=True)
    record = driver.read(reference, tmp_path)
    paid = driver.read(record['evidence_basis']['paid_evidence_ref'], tmp_path)
    model = str(workspace / 'paid_parent.py')
    score = {'model': model, 'scope_ref': save(workspace / 'paid_scope.json', {'rows': paid}),
        'result': {'returncode': 0, 'mse': .25, 'evaluation_scope': 'paid_evaluation',
            'fitting_view': {'scope': 'original_paid_training'}}}
    record['working_comparison'].update(before_model=model, after_model=str(workspace / 'later_candidate.py'),
        before_score_ref=save(workspace / 'paid_before_score.json', score))
    record['theory_target'] = {'parent_binding_obligation': {'source_model': model if retained else None}}
    library, _ = baseline(tmp_path)
    agent = driver.CommonArchiveInquiryAgent(None, tmp_path / 'restored', arm='hegel', source_task=False,
        knowledge={'files': {}, 'theory_consolidation': library},
        config=driver.InquiryLLMConfig(observation_budget=0, max_model_calls=0, token_budget=0,
            execution_seconds=0., compute_budget=100))
    if retained:
        agent.rows = [Observation(row['features'], row['target'], tuple(row['history'])) for row in paid]
        agent.evaluated_model = {'model': model, 'source_model': model, 'current_mse': score['result']['mse'],
            'evaluated_rows': len(paid), 'evaluation_scope': 'paid_evaluation',
            'fitting_view': score['result']['fitting_view']}
        original = driver.begin_method_feedback(agent, driver.AnonymousHistoryLab(kind='hysteresis', seed=1))
        record['profile_ref'] = save(workspace / 'method_feedback/profile.json', original['profile'])
        agent.evaluated_model = {'model': 'later_candidate.py', 'current_mse': 0., 'evaluated_rows': len(paid)}
    return agent, record, score


def test_paid_parent_score_restores_native_pre_call_profile_without_rescoring(tmp_path, monkeypatch):
    agent, record, score = paid_parent_scene(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('restoring a paid scene must not score or execute a model')
    monkeypatch.setattr(agent, '_score', forbidden)
    monkeypatch.setattr(agent, '_execute', forbidden)
    laboratory, same = driver._public_paid_scene(agent, record,
        {'physical_case': {'kind': 'hysteresis', 'seed': 1}})
    assert same is True
    summary = agent._evaluated_summary()
    assert summary['source_model'] == score['model']
    assert summary['mse'] == score['result']['mse'] and summary['evaluated_rows'] == 8
    assert summary['fitting_view'] == score['result']['fitting_view']
    rebuilt = driver.begin_method_feedback(agent, laboratory)['profile']
    assert rebuilt['representation']['kind'] == 'not_exposed'
    assert rebuilt['residual']['state'] == 'unresolved'
    assert rebuilt['residual']['loss'] == score['result']['mse']
    assert agent.costs['compute_cost'] == agent.costs['model_calls'] == 0


def test_comparison_before_model_is_not_a_pre_call_parent_without_recorded_identity(tmp_path):
    agent, record, _ = paid_parent_scene(tmp_path, retained=False)
    laboratory, same = driver._public_paid_scene(agent, record,
        {'physical_case': {'kind': 'hysteresis', 'seed': 1}})
    assert same is True and agent.evaluated_model is None
    rebuilt = driver.begin_method_feedback(agent, laboratory)['profile']
    assert rebuilt['representation']['kind'] == 'unmodeled'
    assert rebuilt['residual']['state'] == 'unmeasured' and rebuilt['residual']['loss'] is None


@pytest.mark.parametrize('wrong_role', ('after_model', 'different_paid_scope'))
def test_pre_call_parent_does_not_borrow_later_model_or_other_paid_rows(tmp_path, wrong_role):
    agent, record, score = paid_parent_scene(tmp_path)
    if wrong_role == 'after_model':
        score['model'] = record['working_comparison']['after_model']
        save(Path(record['working_comparison']['before_score_ref']['file']), score)
    else:
        scope = driver.read(score['scope_ref'], tmp_path)
        scope['rows'].pop()
        save(Path(score['scope_ref']['file']), scope)
    with pytest.raises(ValueError, match='recorded pre-call parent and rows'):
        driver._public_paid_scene(agent, record, {'physical_case': {'kind': 'hysteresis', 'seed': 1}})
