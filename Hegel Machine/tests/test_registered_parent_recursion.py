"""Restore the actual omitted-state shape without scoring or fitting again."""

import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.registered_source_control import (
    prepare_registered_parent_recursion, run_registered_parent_recursion,
)
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program


def reference(path):
    return {'file': str(path), 'json_path': []}


def saved_failure(tmp_path):
    source = tmp_path / 'original'
    agent = CommonArchiveInquiryAgent(None, source, arm='hegel', source_task=False,
        config=InquiryLLMConfig(compute_budget=50000, max_model_calls=0))
    parent = Program('feature', name='u')
    model = 'actual_parent.py'
    (source / model).write_text(export_temporal_program(parent))
    register_theories(agent, [{'id': 'actual_new_parent', 'program': parent.to_dict()}])
    for values in [[.1, .2], [.3, .4], [.5, .6]]:
        rows = [{'features': {'u': value}, 'history': [], 'target': value + .1}
                for value in values]
        offset = len(agent.rows)
        agent.observations.append({'rows': rows, 'row_indices': [offset, offset + 1],
            'action': {'kind': 'trajectory', 'sequence': [{'u': value} for value in values]}})
        agent.rows.extend(Observation(row['features'], row['target'], ()) for row in rows)
    agent._write_evidence()
    evaluated = source / '_evaluated_model'
    evaluated.mkdir()
    (evaluated / model).write_text((source / model).read_text())
    agent._open_question({'id': 'actual_parent_question', 'kind': 'value',
                         'question': 'Predict the paid response.'})
    measured = tmp_path / 'measured_false.json'
    measured.write_text(json.dumps({'finite_status': 'observed_false', 'evidence': [4, 5]}))
    child = agent._open_question({'id': 'actual_revision', 'kind': 'program_revision',
        'question': 'Revise the actual retained predictor after its measured failure.',
        'parent_id': 'actual_parent_question', 'origin': {'row_indices': [4, 5],
                                                       'source_ref': reference(measured)},
        'native_representation_plan': {'parent_theory_ids': ['actual_new_parent'],
            'validation_records': [1], 'compute_limit': 5000, 'max_candidates': 4},
        'native_representation_research': [{'status': 'native_representation_execution_failed',
                                           'candidate_count': 0}]})
    agent._active_question_id = child['id']
    agent.costs.update(compute_cost=99, python_execution_seconds=2.)
    score = {'returncode': 0, 'mse': .01, 'paid_rows': 6,
             'representation_diagnostic': {'status': 'no_exact_conflicts'}}
    agent.research_candidates[model] = {'current_evaluation': score, 'paid_rows': 6,
                                       'status': 'evaluated'}
    agent.evaluated_model = {'model': model, 'source_model': model,
                            'current_mse': .01, 'evaluated_rows': 6}
    result = tmp_path / 'failed_result.json'
    result.write_text(json.dumps(agent.checkpoint_result('failed_retained_no_retry')))
    assert not (source / 'research_state.json').exists()
    initial = tmp_path / 'initial.json'
    initial.write_text(json.dumps(score))
    roster = tmp_path / 'roster.json'
    roster.write_text(json.dumps([str(source / model)]))
    scope = tmp_path / 'old_scope.json'
    scope.write_text(json.dumps({'rows': [1, 2]}))
    index = tmp_path / 'assessments.json'
    index.write_text(json.dumps({'scopes': [reference(scope)], 'assessments': [], 'contexts': []}))
    procedure = tmp_path / 'procedure.json'
    procedure.write_text(json.dumps({'program': parent.to_dict(),
                                     'research_program': {'graph': 'retained_by_reference'}}))
    return {'execution_admitted': False, 'source_result_ref': reference(result),
        'source_initial_evaluation_ref': reference(initial), 'source_workspace': str(source),
        'question_id': child['id'], 'parent_theory_id': 'actual_new_parent',
        'counterexample_rows': [4, 5], 'starting_costs': {**agent.costs, 'compute_cost': 199},
        'python_cost_basis': 'conservative_upper_bound_not_exact_external_actual',
        'limits': {'compute': 50000, 'python_seconds': 30., 'native_unit_compute': 5000,
                   'max_candidates': 4},
        'environment': {'kind': 'hysteresis', 'seed': 13},
        'frozen_split': {'training_rows': 4, 'validation_rows': 2, 'validation_records': [1]},
        'retained_roster_ref': reference(roster), 'complete_old_pool_count': 1,
        'retained_score_index_ref': reference(index),
        'retained_research_program_refs': [reference(procedure)],
        'condition_negative_ref': {'file': 'prior_not_separable.json', 'json_path': []},
        'prior_cost_refs': [reference(result)]}


def test_failed_scene_cold_restore_preserves_paid_score_and_missing_state_is_only_written_locally(
        tmp_path, monkeypatch):
    registration = saved_failure(tmp_path)
    original = json.loads((tmp_path / 'failed_result.json').read_text())

    def forbidden(*args, **kwargs):
        raise AssertionError('restoration repeated paid scientific work')

    monkeypatch.setattr(CommonArchiveInquiryAgent, '_score', forbidden)
    monkeypatch.setattr(CommonArchiveInquiryAgent, '_python', forbidden)
    scene = prepare_registered_parent_recursion(registration, tmp_path / 'prepared')
    accepted = scene['acceptance']
    assert accepted['new_predictions'] == accepted['new_fits'] == accepted['new_observations'] == 0
    assert accepted['source_questions_preserved'] and accepted['source_parent_bytes_equal']
    assert accepted['parent_mse'] == .01
    assert accepted['initial_parent_score_reused_without_evaluation']
    assert accepted['source_reported_costs'] == original['costs']
    assert scene['agent'].costs == registration['starting_costs']
    assert scene['agent'].questions[-1]['native_representation_plan'][
        'retained_research_program_refs'] == registration['retained_research_program_refs']
    assert not (tmp_path / 'original/research_state.json').exists()
    assert (tmp_path / 'prepared/workspace/research_state.json').is_file()
    assert not accepted['lost_intermediate_fits_recovered']
    retained_index = json.loads((scene['agent'].workspace
        / '_paid_python_scores/_evaluated_model/assessments.json').read_text())
    assert retained_index['scopes'] == [reference(tmp_path / 'old_scope.json')]
    assert json.loads((tmp_path / 'failed_result.json').read_text()) == original


def test_unadmitted_recursive_run_stops_before_creating_or_loading_scene(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unadmitted run reached restoration')

    monkeypatch.setattr('hegel_machine.cognition.registered_source_control.'
                        'prepare_registered_parent_recursion', forbidden)
    with pytest.raises(ValueError, match='not admitted'):
        run_registered_parent_recursion({'execution_admitted': False}, tmp_path / 'not_created')
    assert not (tmp_path / 'not_created').exists()


def test_restore_rejects_a_procedure_from_another_parent_before_writing_bridge(tmp_path):
    registration = saved_failure(tmp_path)
    procedure = registration['retained_research_program_refs'][0]['file']
    value = json.loads(open(procedure).read())
    value['program'] = Program('const', value=0.).to_dict()
    with open(procedure, 'w') as stream:
        json.dump(value, stream)
    with pytest.raises(ValueError, match='actual saved parent'):
        prepare_registered_parent_recursion(registration, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected/retained_scene').exists()


def test_external_timing_counts_nested_fresh_calls_once_and_ignores_retained_cost():
    from hegel_machine.cognition.registered_source_control import _comparison_external_python_timing

    fresh = {'external_python_wall_seconds': 2., 'paid_score_reuse': {
        'status': 'newly_scored', 'execution_basis_ref': reference('assessment.json')}}
    alias = {'paid_score_reuse': {'status': 'newly_scored',
        'assessment_ref': {'file': 'assessment.json', 'json_path': ['result']}},
        'external_python_wall_seconds': 2.}
    old = {'external_python_wall_seconds': 90., 'paid_score_reuse': {'status': 'reused'},
           'retained_comparison': fresh}
    actual = _comparison_external_python_timing({'evaluations': [fresh, alias, old]})
    assert actual['external_python_wall_seconds'] == 2.
    assert actual['unmeasured_execution_count'] == 0


def test_unmetered_fresh_execution_stays_unknown():
    from hegel_machine.cognition.registered_source_control import _comparison_external_python_timing

    actual = _comparison_external_python_timing({'evaluations': [
        {'paid_score_reuse': {'status': 'newly_scored'}},
        {'paid_score_reuse': {'status': 'reused'}}]})
    assert actual['external_python_wall_seconds'] is None
    assert actual['measured_external_python_wall_seconds'] == 0.
    assert actual['unmeasured_execution_count'] == 1


def test_interrupted_comparison_does_not_present_completed_calls_as_exact_total():
    from hegel_machine.cognition.registered_source_control import _comparison_external_python_timing

    actual = _comparison_external_python_timing({'truncated': True, 'evaluations': [
        {'external_python_wall_seconds': 2., 'paid_score_reuse': {'status': 'newly_scored'}}]})
    assert actual['external_python_wall_seconds'] is None
    assert actual['measured_external_python_wall_seconds'] == 2.
    assert actual['comparison_incomplete']


def test_recursive_stage_keeps_native_host_and_billed_python_separate(tmp_path, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    from scripts import run_anonymous_revision_inquiry as evaluator
    from hegel_machine.cognition import registered_source_control as control

    before, after = Program('const', value=0.), Program('const', value=1.)
    workspace = tmp_path / 'workspace'
    (workspace / '_evaluated_model').mkdir(parents=True)
    for name, program in [('before.py', before), ('after.py', after)]:
        code = export_temporal_program(program)
        (workspace / name).write_text(code)
        (workspace / '_evaluated_model' / name).write_text(code)
    clock = [0.]
    monkeypatch.setattr(control, 'monotonic', lambda: clock[0])
    current = {'source_model': 'before.py', 'model': '_evaluated_model/before.py', 'mse': .1}
    costs = {'compute_cost': 10, 'python_execution_seconds': 5.}

    def execute(action, arguments, laboratory):
        if arguments['operator'] == 'research_route':
            clock[0] += 100.
            return {'native_representation_ref': reference('native.json'),
                    'exported_models': [{'model': 'after.py'}]}
        assert arguments['operator'] == 'research_choice'
        clock[0] += 10.
        costs['python_execution_seconds'] += 10.
        current.update(source_model='after.py', model='_evaluated_model/after.py', mse=.05)
        return {'models_requested': 2, 'models_completed': 2, 'evaluations': [
            {'external_python_wall_seconds': 4.,
             'paid_score_reuse': {'status': 'newly_scored'}}]}

    agent = SimpleNamespace(workspace=workspace, rows=[object(), object()], costs=costs,
        theory_index={'actual': {'program': before.to_dict()}}, candidate_paths=['after.py'],
        config=SimpleNamespace(execution_seconds=1000.), _execute=execute,
        _evaluated_summary=lambda: deepcopy(current), _remaining_compute=lambda: 1000,
        _prepare_program_state=lambda: None,
        checkpoint_result=lambda status: {'status': status, 'costs': deepcopy(costs)})
    agent._charge = lambda amount: costs.update(compute_cost=costs['compute_cost'] + amount)
    native = {'native_host_seconds_before_persistence': 99.,
        'candidates': [{'program': after.to_dict()}], 'parent_theory_ids': ['actual'],
        'origin': {'row_indices': [0]}}
    monkeypatch.setattr('hegel_machine.cognition.representation_inquiry.'
                        'load_native_representation_result', lambda *args: native)

    def forecast(*args, **kwargs):
        clock[0] += 1.
        return {'returncode': 0, 'python_execution_seconds': 1.,
                'external_python_wall_seconds': .25}

    monkeypatch.setattr(evaluator, 'evaluate_terminal_model', forecast)
    scene = {'agent': agent, 'laboratory': None, 'question': {'id': 'revision'},
        'starting_costs': deepcopy(costs), 'models': ['before.py'], 'output': tmp_path}
    registration = {'execution_admitted': True, 'source_result_ref': reference('old.json'),
        'prior_cost_refs': [], 'parent_theory_id': 'actual', 'counterexample_rows': [0]}
    result = run_registered_parent_recursion(registration, tmp_path, accepted_scene=scene)
    assert result['status'] == 'completed'
    assert result['timing']['native_route']['wall_seconds'] == 100.
    assert result['timing']['native_route']['research_body_host_seconds'] == 99.
    assert result['timing']['native_route']['python_charge_seconds'] == 0.
    comparison = result['timing']['strong_pool_comparison']
    assert comparison['python_charge_seconds'] == 10.
    assert comparison['external_python_wall_seconds'] == 4.
    assert comparison['charged_non_executor_seconds'] == 6.
    assert result['new_costs']['python_execution_seconds'] == 12.
    assert result['wall_seconds'] == 112.
