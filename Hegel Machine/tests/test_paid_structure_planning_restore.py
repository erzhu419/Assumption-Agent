"""A paid planning delta returns through the original retained-scene factory."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.native_condition_research import native_structure_planning_contract
from scripts.prepare_pr3_source_successor import (
    _restore_paid_structure_planning, make_retained_method_pairing_factories,
)
from test_native_structural_discrimination import structural_scene


def planning_scene(tmp_path):
    public, agent, laboratory, question, previous, _ = structural_scene()
    training = {'file': str(tmp_path / 'original/training.json'), 'json_path': []}
    state = str(tmp_path / 'original/research_state.json')
    owner = {'question_id': question['id'], 'meaning_version': 1}
    source = {'training_ref': training, 'planning_question_ref': owner}
    basis = native_structure_planning_contract(public, agent.observations, previous)
    prefix = ['questions', 0, 'native_condition_research', 0]
    target = {'source_question_ref': deepcopy(owner), 'source_round_index': 0,
        'hypothesis_refs': [{'file': state, 'json_path': [*prefix, 'candidates', index]}
                            for index in basis['candidate_indices']]}
    # Frozen synthetic metadata only: no planner or predictor is run here.
    planning = {'basis': basis, 'source_ref': {'file': state, 'json_path': prefix},
        'source_round_index': 0, 'planning_index': 0, 'question_meaning_version': 1,
        'planning_complete': True, 'compute_cost': 168,
        'proposed_experiments': [{'structural_discrimination': target}]}
    agent.evaluated_model = {'model': 'candidate_0165.py', 'current_mse': .1, 'evaluated_rows': 12}
    return public, agent, laboratory, question, source, planning


def test_paid_delta_restores_once_without_prediction_or_recharging(tmp_path, monkeypatch):
    _, agent, lab, question, source, planning = planning_scene(tmp_path)
    monkeypatch.setattr(Program, 'predict', lambda *_: pytest.fail('restore ran a scientific prediction'))
    before = deepcopy((agent.costs, agent.evaluated_model, agent.observations, agent.candidate_paths))
    returned = _restore_paid_structure_planning(agent, lab, source, planning)
    assert returned is question and question['native_structure_planning'] == [planning]
    assert question['native_structure_planning'][0] is not planning
    _restore_paid_structure_planning(agent, lab, source, planning)
    assert len(question['native_structure_planning']) == 1
    assert (agent.costs, agent.evaluated_model, agent.observations, agent.candidate_paths) == before


@pytest.mark.parametrize('change', ['meaning', 'bank', 'evidence', 'source', 'candidate_reference'])
def test_wrong_paid_planning_scope_is_not_rebound(tmp_path, change):
    _, agent, lab, question, source, planning = planning_scene(tmp_path)
    if change == 'meaning':
        source['planning_question_ref']['meaning_version'] = 2
    elif change == 'bank':
        planning['basis']['source_round_index'] = 1
    elif change == 'evidence':
        planning['basis']['observed_record_count'] += 1
    elif change == 'source':
        planning['source_ref']['file'] = str(tmp_path / 'other/research_state.json')
    else:
        planning['proposed_experiments'][0]['structural_discrimination']['hypothesis_refs'][0][
            'json_path'][-1] = 999
    before = deepcopy(agent.costs)
    with pytest.raises(ValueError, match='paid structur'):
        _restore_paid_structure_planning(agent, lab, source, planning)
    assert not question.get('native_structure_planning') and agent.costs == before


def test_current_factory_attaches_paid_plan_before_retained_parent_branch(tmp_path, monkeypatch):
    from hegel_machine.cognition import experiment_runtime, registered_source_scene

    _, agent, lab, question, source, planning = planning_scene(tmp_path)
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'file': str(path), 'json_path': []}
    costs = {'compute_cost': 1040, 'observation_cost': 12, 'measurement_cost': 0,
        'python_execution_seconds': 2., 'model_calls': 0, 'reported_tokens': 0}
    source.update(case_id='case_002', paid_observations_ref={**source['training_ref'],
        'json_path': ['paid_observations']}, costs_ref=save('costs.json', costs),
        initial_archive_ref=save('archive.json', {'files': {}}),
        plan_ref=save('plan.json', {'limits': {}, 'cases': [{'id': 'case_002'}]}),
        planning_result_ref=save('paid_plan.json', planning),
        unknown_cost_ref={'file': 'retained_unknown_costs.json'})
    agent.costs = deepcopy(costs)
    agent.workspace = tmp_path / 'restored'
    agent._evaluated_summary = lambda: {'model': '_evaluated_model/candidate_0165.py'}
    scene = {'agent': agent, 'lab': lab, 'models': ['old.py']}
    def restored(*args, **kwargs):
        assert kwargs['starting_costs'] == costs
        assert kwargs['limits'].compute_budget == 1060
        return scene
    monkeypatch.setattr(registered_source_scene, 'restore_retained_common_scene', restored)
    monkeypatch.setattr(experiment_runtime, '_initialize_worker', lambda *_: None)
    monkeypatch.setattr(experiment_runtime, 'physical_cpu_groups', lambda *_: [[0]])
    monkeypatch.setattr(Program, 'predict', lambda *_: pytest.fail('restore predicted a value'))
    contract = {'start_state_ref': source['training_ref'], 'data_ref': source['paid_observations_ref'],
        'compute_cap': 20, 'observation_cap': 0, 'time_limits': {'billed_python_seconds': 30.},
        'scope': {'paid_rows': 12, 'pool_models': 1, 'max_exports': 0},
        'execution_role': 'retained_parent_baseline', 'retained_parent': {
            'model': '_evaluated_model/candidate_0165.py',
            'source_ref': {'file': str(tmp_path / 'original/research_state.json'),
                'json_path': ['evaluated_model']},
            'author_work_ref': {'file': 'original_p0_choice.json'}},
        **{key: {'file': key + '.json'} for key in (
            'split_ref', 'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}
    callbacks = make_retained_method_pairing_factories({'source_scene': source,
        'execution_authorized': True, 'approval_ref': {'file': 'synthetic_approval.json'},
        'maximum_physical_cores': 1}, tmp_path / 'output')
    actual = callbacks['restore_cell']('00', contract, tmp_path / 'output/00')
    assert actual['path'] == '_evaluated_model/candidate_0165.py'
    assert actual['planning_result_ref'] == source['planning_result_ref']
    assert question['native_structure_planning'] == [planning]
    assert agent.costs == costs and actual['models'] == ['old.py']


@pytest.mark.parametrize('change,narrow', [
    (None, True), ('paid_export', True), ('validated_export', True),
    ('predictor_alias', True), ('paid_export', False),
])
@pytest.mark.parametrize('resume', [False, True])
def test_paid_planning_pool_is_checked_after_public_work_before_policy(tmp_path, monkeypatch, change, narrow, resume):
    from hegel_machine.cognition import experiment_runtime, method_feedback, registered_source_scene
    from hegel_machine.cognition import research_programs, theory_consolidation
    from hegel_machine.cognition.finite_native_author import _comparison_models

    _, agent, lab, question, source, planning = planning_scene(tmp_path)
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'file': str(path), 'json_path': []}
    costs = {'compute_cost': 1040, 'observation_cost': 12, 'measurement_cost': 0,
        'python_execution_seconds': 2., 'model_calls': 0, 'reported_tokens': 0}
    source.update(case_id='case_002', paid_observations_ref={**source['training_ref'],
        'json_path': ['paid_observations']}, costs_ref=save('costs.json', costs),
        initial_archive_ref=save('archive.json', {'files': {}}),
        plan_ref=save('plan.json', {'limits': {}, 'cases': [{'id': 'case_002'}]}),
        planning_result_ref=save('paid_plan.json', planning),
        unknown_cost_ref={'file': 'retained_unknown_costs.json'})
    agent.costs = deepcopy(costs)
    agent.workspace = tmp_path / 'restored'
    models = _comparison_models(agent.knowledge['finite_native_author']['predictor_paths'],
        agent.candidate_paths, question['native_condition_research'])
    # Real restore gives both views this same complete roster. The guard must
    # freeze its tuple before a before-hook can append through either alias.
    agent.knowledge['finite_native_author']['predictor_paths'] = models
    scene = {'agent': agent, 'lab': lab, 'models': models}
    agent._organization_state = {'registered_method_unit': {'phase': 'research',
        'current_scope': {'question_id': question['id'], 'meaning_version': 1}},
        'finite_author': {'completed_steps': 0}}
    trace = []
    def public_work(_lab):
        trace.append('public_work')
        if change == 'paid_export':
            agent.candidate_paths.append('paid_export.py')
        elif change == 'validated_export':
            agent.operator_results.append({'operator': 'refit',
                'status': 'refit_supported_on_validation', 'path': 'paid_export.py'})
        elif change == 'predictor_alias':
            agent.knowledge['finite_native_author']['predictor_paths'].append('paid_export.py')
        return True
    agent._before_model_turn = public_work
    agent._execute = lambda *_: pytest.fail('query, native or scoring occurred before scope admission')
    agent._save = lambda *_: None
    candidate = {'id': 'P1', 'version': 1, 'execution': {'path': 'shared/P1.py'}}
    monkeypatch.setattr(registered_source_scene, 'restore_retained_common_scene', lambda *a, **k: scene)
    monkeypatch.setattr(experiment_runtime, '_initialize_worker', lambda *_: None)
    monkeypatch.setattr(experiment_runtime, 'physical_cpu_groups', lambda *_: [[0]])
    monkeypatch.setattr(theory_consolidation, 'install_python_research_method', lambda *a, **k: candidate)
    monkeypatch.setattr(theory_consolidation, 'library_entry_reference', lambda *a, **k: {
        'file': 'shared/theory_library.json', 'json_path': ['candidates', 'P1']})
    monkeypatch.setattr(research_programs, 'export_research_selection_program', lambda *_: '# synthetic policy')
    monkeypatch.setattr(method_feedback, 'begin_method_feedback', lambda *a, **k: {'profile': {}})
    monkeypatch.setattr(method_feedback, '_profile_ref', lambda *a, **k: {'file': 'profile.json'})
    contract = {'start_state_ref': source['training_ref'], 'data_ref': source['paid_observations_ref'],
        'compute_cap': 20, 'observation_cap': 0, 'time_limits': {'billed_python_seconds': 30.},
        'scope': {'paid_rows': 12, 'pool_models': len(models), 'max_exports': 0},
        'selection_plan': {'ordinary_method_path': 'shared/P0.py'},
        'method_ref': {'id': 'P1', 'version': 1}, 'method_source_refs': [], 'max_steps': 1,
        **{key: {'file': key + '.json'} for key in (
            'split_ref', 'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}
    registration = {'source_scene': source, 'execution_authorized': True,
        'approval_ref': {'file': 'synthetic_approval.json'}, 'maximum_physical_cores': 1}
    if resume:
        from scripts import run_method_action_pairing as runner
        failed = {**source['training_ref'], 'file': str(tmp_path / 'failed/training.json')}
        registration['continuation'] = {'outcome_refs': {'10': {'file': 'failed_outcome.json'}}}
        monkeypatch.setattr(runner, '_pair_continuation', lambda *_: {'outcomes': {'10': {
            'continuation_state_ref': failed,
            'continuation_data_ref': {**failed, 'json_path': ['paid_observations']},
            'continuation_costs_ref': source['costs_ref']}}})
    if narrow:
        registration['comparison_kind'] = 'additional_research_vs_retained_parent'
    actual = make_retained_method_pairing_factories(registration, tmp_path / 'output')[
        'restore_cell']('10', contract, tmp_path / 'output/10')
    if resume:
        assert actual['physical_resume_state_ref'] == failed
        assert actual['prior_outcome_ref'] == registration['continuation']['outcome_refs']['10']
        assert actual['start_state_ref'] == source['training_ref']
    if change and narrow:
        with pytest.raises(ValueError, match='pool changed.*re-quote'):
            actual['before_step'](lab)
        assert trace == ['public_work']
        assert agent._organization_state['registered_method_unit']['phase'] == 'requires_rebind'
        assert ('paid_export.py' in agent.candidate_paths or change == 'validated_export'
                or 'paid_export.py' in agent.knowledge['finite_native_author']['predictor_paths'])
    else:
        assert actual['before_step'](lab) is True
        trace.append('policy_selects_planned_query')
        assert trace == ['public_work', 'policy_selects_planned_query']
    assert agent.costs == costs
