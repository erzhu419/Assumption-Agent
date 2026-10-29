"""The explicit frozen parent is evaluable without inventing a P0 research step."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_theory_consolidation import agent_scene
from test_v8_method_action_pairing import registration, runner


def retained_registration():
    plan = registration()
    plan['comparison_kind'] = 'additional_research_vs_retained_parent'
    plan['components']['ordinary_use'] = {'compute': 0, 'observations': 0}
    baseline = plan['cells']['00']
    baseline.update(execution_role='retained_parent_baseline', retained_parent={
        'model': '_evaluated_model/parent.py',
        'source_ref': {'file': 'paid/research_state.json', 'json_path': ['evaluated_model']},
        'research_status': 'author_work_required',
        'author_work_ref': {'file': 'actual157_choice.json', 'json_path': ['P0']}})
    for key in ('search', 'export', 'complete_comparison'):
        baseline['components'][key] = {'compute': 0, 'observations': 0}
    plan['cells']['10']['scope']['max_exports'] = 1
    plan.update(execution_authorized=True, approval_ref={'file': 'approved_control.json'})
    plan['quote'] = runner.quote_method_action_pairing(plan)
    assert plan['quote']['status'] == 'quoted'
    return plan


def current_parent(agent):
    code = export_temporal_program(Program('const', value=1.), feature_names=('u',))
    agent._save('old.py', code)
    agent._save('_evaluated_model/parent.py', code)
    agent.evaluated_model = {'model': 'parent.py', 'source_model': 'old.py',
        'current_mse': 1., 'evaluated_rows': 1}


@pytest.mark.parametrize('native_returned', [True, False])
def test_declared_parent_freezes_without_research_but_challenger_still_requires_native(
        tmp_path, monkeypatch, native_returned):
    plan = retained_registration()
    agents, evaluated, prepared = {}, [], []

    def restore(name, contract, output):
        agent, lab = agent_scene(output)
        lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
            'max_probe_length': 2, 'feature_bounds': {'u': [-1., 1.]}}
        agent.rows = [Observation({'u': 0.}, 2.)]
        current_parent(agent)
        agent._save('method.py', 'def select_actions(*args): return {"stop": True}\n')
        agent._open_question({'id': 'q', 'question': 'Synthetic current parent', 'parent_id': 'task'})
        agents[name] = agent
        if name == '00':
            agent._execute = lambda *_: pytest.fail('baseline may not rescore or research')
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': 'q'}, 'native_attempt_limit': 1,
            'profile_ref': None, 'origin_ref': contract.get('retained_parent', {}).get('author_work_ref'),
            **{key: contract[key] for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    def method(agent, *args, **kwargs):
        assert agent is agents['10'], 'P0 is a declared retained predictor, not a method invocation'
        if native_returned:
            agent._save('new.py', export_temporal_program(Program('const', value=2.), feature_names=('u',)))
            agent.candidate_paths.append('new.py')
            agent.operator_results.append({'operator': 'native_condition_research',
                'status': 'candidates_require_evaluation', 'native_condition_research': {
                    'status': 'candidates_require_evaluation', 'structural_discrimination': {
                        'forecast_verdict': {'complete': True, 'next_research': {
                            'kind': 'jointly_refit_frozen_structures'}},
                        'refit_return': {'status': 'candidate_supported_on_paid_rows'}}}})
        return {'status': 'completed' if native_returned else 'author_work_required',
            'model': 'new.py' if native_returned else None,
            'registered_attempt_returned': native_returned, 'actual_action_content': [],
            'native_boundary_refs': ([{'file': str(agent.workspace / 'research_state.json'),
                'json_path': ['operator_results', 0]}] if native_returned else [])}
    monkeypatch.setattr(runner, 'execute_registered_method_unit', method)

    def prepare():
        manifest = json.loads((tmp_path / 'study/frozen_outputs.json').read_text())
        assert set(manifest['models']) == {'00', '10'}
        prepared.append(True)
        return {'file': 'synthetic_common_labels.json'}

    def evaluate(name, frozen, reference, account):
        evaluated.append(name)
        namespace = {}
        exec((Path(frozen['workspace']) / frozen['model']).read_text(), namespace)
        value = namespace['predict']({'features': {'u': 0.}, 'history': []})
        account.costs['compute_cost'] += 1
        loss = (value - 2.) ** 2
        return {'loss': loss, 'quality': {'loss': loss, 'threshold': 1e-10},
            'completed': loss <= 1e-10, 'coverage': 1}

    result = runner.run_method_action_pairing(plan, tmp_path / 'study', restore_cell=restore,
        evaluate_frozen=evaluate, prepare_evaluation=prepare,
        ordinary_use=lambda *_: pytest.fail('retained-parent contrast is not ordinary-method support'))
    baseline = json.loads(Path(result['outcome_refs']['00']['file']).read_text())
    challenger = json.loads(Path(result['outcome_refs']['10']['file']).read_text())
    assert baseline['execution_complete'] and not baseline['research_execution_complete']
    assert baseline['research_status'] == 'author_work_required'
    assert baseline['author_work_ref'] == plan['cells']['00']['retained_parent']['author_work_ref']
    assert baseline['method_delivery'] == 'registered_retained_parent_baseline'
    assert baseline['costs']['compute_cost'] == int(native_returned)
    assert baseline['continuation_scope']['pool_models'] == 1
    assert result['ordinary_use']['status'] == 'not_admitted_retained_parent_control'
    assert challenger['execution_complete'] is native_returned
    assert prepared == [True] if native_returned else prepared == []
    assert evaluated == ['00', '10'] if native_returned else evaluated == []
    pair = json.loads(Path(result['pair_refs'][0]['file']).read_text())
    assert pair['comparison_kind'] == 'additional_research_vs_retained_parent'
    if native_returned:
        assert pair['frozen_predictive_comparison']['challenger_minus_baseline'] == -1.
        assert pair['baseline_research_status'] == 'author_work_required'
    else:
        assert challenger['method_delivery'] == 'preparation_only'
        assert not result['evaluation_lifecycle']['labels_revealed']


def test_parent_declaration_is_in_quote_and_different_current_parent_is_not_substituted(tmp_path):
    plan = retained_registration()
    changed = deepcopy(plan)
    changed['cells']['00']['retained_parent']['model'] = '_evaluated_model/another.py'
    assert runner.quote_method_action_pairing(changed) != plan['quote']
    agent, _ = agent_scene(tmp_path)
    agent.rows = [Observation({'u': 0.}, 2.)]
    current_parent(agent)
    before = deepcopy(agent.costs)
    with pytest.raises(ValueError, match='differs from the registered'):
        runner._registered_retained_parent(agent, changed['cells']['00'])
    assert agent.costs == before


@pytest.mark.parametrize('change', ['no_scope', 'no_identity', 'research_fees', 'ordinary_use'])
def test_unregistered_parent_bypass_or_hidden_research_is_rejected(change):
    plan = retained_registration()
    if change == 'no_scope':
        plan.pop('comparison_kind')
    elif change == 'no_identity':
        plan['cells']['00'].pop('retained_parent')
    elif change == 'research_fees':
        plan['cells']['00']['components']['search']['compute'] = 1
    else:
        plan['components']['ordinary_use']['compute'] = 1
    assert runner.quote_method_action_pairing(plan)['status'] == 'quote_incomplete'


@pytest.mark.parametrize('returned_status,expected', [
    ('frozen_predictions_compared', True), ('frozen_readout_incomplete', False),
    ('search_incomplete', False), ('candidates_require_evaluation', True), ('error', False)])
def test_actual_native_return_distinguishes_readout_completion_from_a_paid_attempt(
        tmp_path, monkeypatch, returned_status, expected):
    plan = retained_registration()

    def restore(name, contract, output):
        agent, lab = agent_scene(output)
        lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
            'max_probe_length': 2, 'feature_bounds': {'u': [-1., 1.]}}
        agent.rows = [Observation({'u': 0.}, 2.)]
        current_parent(agent)
        agent._save('method.py', 'def select_actions(*args): return {"stop": True}\n')
        agent._open_question({'id': 'q', 'question': 'Synthetic current parent', 'parent_id': 'task'})
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': 'q'}, 'native_attempt_limit': 1,
            'profile_ref': None, 'origin_ref': None,
            **{key: contract[key] for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    def procedure(agent, lab, _path, **kwargs):
        kwargs['before_step'](lab)
        joint = returned_status in {'search_incomplete', 'candidates_require_evaluation'}
        work = {'status': returned_status, 'structural_discrimination': {
            'forecast_verdict': {'complete': returned_status != 'frozen_readout_incomplete',
                'next_research': {'kind': 'jointly_refit_frozen_structures' if joint else 'retain_frozen_prediction'}},
            **({'refit_return': {'status': 'computation_incomplete' if returned_status == 'search_incomplete'
                else 'candidate_supported_on_paid_rows'}} if joint else {})}}
        feedback = {'operator': 'native_condition_research', 'status': returned_status,
            'native_condition_research': work}
        if returned_status == 'error':
            feedback['error'] = 'Synthetic actual native failure'
        agent.operator_results.append(feedback)
        agent.method_results.append({'path': 'method.py', 'proposal': {'action': 'run',
            'arguments': {'operator': 'native_condition_research'}}, 'feedback': deepcopy(feedback)})
        kwargs['after_step'](lab)
        pytest.fail('the existing native-return boundary must interrupt this procedure')
    monkeypatch.setattr(runner, 'execute_procedure', procedure)
    evaluations = []
    result = runner.run_method_action_pairing(plan, tmp_path / 'study', restore_cell=restore,
        evaluate_frozen=lambda *args: evaluations.append(args[0]) or {
            'loss': 1., 'quality': {'loss': 1., 'threshold': 1e-10}, 'coverage': 1, 'completed': False},
        ordinary_use=lambda *_: pytest.fail('narrow control cannot authorize method use'))
    challenger = json.loads(Path(result['outcome_refs']['10']['file']).read_text())
    assert challenger['registered_attempt_returned']
    assert challenger['research_return_verification']['complete'] is expected
    assert challenger['execution_complete'] is expected
    assert bool(evaluations) is expected
    if returned_status == 'frozen_predictions_compared':
        actual = challenger['research_return_verification']['actual_returns'][0]
        assert actual['frozen_readout_complete'] and not actual['joint_refit_required']
    if not expected:
        assert challenger['method_delivery'] == 'research_return_incomplete'
        assert challenger['comparison']['models_requested'] is None
        assert not result['evaluation_lifecycle']['labels_revealed']


@pytest.mark.parametrize('same_source', [True, False])
def test_production_factory_binds_literal_checkpoint_without_installing_a_method(
        tmp_path, monkeypatch, same_source):
    from scripts.prepare_pr3_source_successor import make_retained_method_pairing_factories
    from hegel_machine.cognition import experiment_runtime, method_feedback
    from hegel_machine.cognition import registered_source_scene

    def saved(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'file': str(path), 'json_path': []}

    plan = retained_registration()
    costs = {'compute_cost': 7, 'observation_cost': 1, 'measurement_cost': 0,
        'python_execution_seconds': 2., 'model_calls': 0, 'reported_tokens': 0}
    training = {'file': str(tmp_path / 'paid/training.json'), 'json_path': []}
    source = {'case_id': 'case_002', 'training_ref': training,
        'paid_observations_ref': {**training, 'json_path': ['paid_observations']},
        'costs_ref': saved('costs.json', costs), 'unknown_cost_ref': {'file': 'old_unknown.json'},
        'plan_ref': saved('source_plan.json', {'limits': {}, 'cases': [{'id': 'case_002'}]}),
        'initial_archive_ref': saved('archive.json', {'files': {}, 'finite_native_author': {
            'question': {'id': 'q'}}})}
    contract = plan['cells']['00']
    contract['start_state_ref'] = training
    contract['data_ref'] = source['paid_observations_ref']
    contract['retained_parent']['source_ref'] = {
        'file': str(tmp_path / ('paid' if same_source else 'other') / 'research_state.json'),
        'json_path': ['evaluated_model']}
    plan.update(source_scene=source, maximum_physical_cores=1)

    def restore(source_path, workspace, **kwargs):
        agent, lab = agent_scene(workspace)
        agent.rows = [Observation({'u': 0.}, 2.)]
        agent.knowledge = kwargs['knowledge']
        agent.costs = deepcopy(kwargs['starting_costs'])
        agent.config = kwargs['limits']
        current_parent(agent)
        agent._save = lambda *_: pytest.fail('the baseline must not install or rewrite a method')
        return {'agent': agent, 'lab': lab, 'models': ['old.py']}
    monkeypatch.setattr(registered_source_scene, 'restore_retained_common_scene', restore)
    monkeypatch.setattr(experiment_runtime, '_initialize_worker', lambda _: None)
    monkeypatch.setattr(experiment_runtime, 'physical_cpu_groups', lambda *_: [[0]])
    monkeypatch.setattr(method_feedback, 'begin_method_feedback',
        lambda *_a, **_kw: pytest.fail('a frozen parent is not a new method use'))
    callbacks = make_retained_method_pairing_factories(plan, tmp_path / 'output')
    if same_source:
        scene = callbacks['restore_cell']('00', contract, tmp_path / 'output/00')
        assert scene['path'] == contract['retained_parent']['model']
        assert scene['profile_ref'] is None and scene['agent'].costs == costs
        assert scene['native_attempt_limit'] == 1
    else:
        with pytest.raises(ValueError, match='literal parent at this registered source'):
            callbacks['restore_cell']('00', contract, tmp_path / 'output/00')
