"""Frozen development-cell orchestration, not scientific-effect experiments."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_theory_consolidation import agent_scene

spec = importlib.util.spec_from_file_location('v8_pair_runner',
    Path(__file__).resolve().parents[1] / 'scripts/run_method_action_pairing.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def registration(*, four=False):
    component = {'compute': 10, 'observations': 0}
    cells = {}
    for name in ('00', '10', '01', '11') if four else ('00', '10'):
        plus = name.endswith('1')
        cells[name] = {'method_path': 'method.py', 'scope': {'paid_rows': 2 if plus else 1,
            'pool_models': 1, 'max_exports': 0}, 'components': {
                key: deepcopy(component) for key in runner.CELL_COMPONENTS},
            **{key + '_ref': {'file': key + ('1' if plus and key in {'start_state', 'data'} else '0') + '.json',
                             'json_path': []} for key in ('start_state', 'data', 'split',
                                 'quality_contract', 'resource_contract', 'evaluation')},
            'return_contract': 'retained_parent', 'time_limits': {
                'billed_python_seconds': 30, 'active_wall_seconds': 60}, 'max_steps': 2,
            'compute_cap': 100, 'observation_cap': 0,
            'action_content': {'operator': 'old' if name[0] == '0' else 'alternative'}}
        cells[name]['method_ref'] = {'id': 'old-method' if name[0] == '0' else 'alternative-method',
                                    'version': 1}
    return {'execution_authorized': False, 'approval_ref': None, 'cells': cells,
        'components': {key: {'compute': 10, 'observations': 0} for key in runner.GLOBAL_COMPONENTS},
        'D1': {'legal': four, 'proposal_ref': {'file': 'D1.json', 'json_path': []} if four else None},
        'method_ref': {'id': 'method', 'version': 1}, 'compute_cap': 10000, 'observation_cap': 0}


def test_n_or_k_change_invalidates_quote_and_unknown_component_is_not_zero(tmp_path):
    plan = registration()
    quoted = runner.quote_method_action_pairing(plan)
    assert quoted['status'] == 'quoted'
    scopes = deepcopy(quoted['scopes']); scopes['00']['max_exports'] += 1
    assert runner.quote_method_action_pairing(plan, current_scopes=scopes)['status'] == 'quote_incomplete'
    plan['components']['prepare']['compute'] = None
    assert runner.quote_method_action_pairing(plan)['compute'] is None
    called = []
    with pytest.raises(ValueError):
        runner.run_method_action_pairing(plan, tmp_path,
            restore_cell=lambda *args: called.append(True), evaluate_frozen=None, ordinary_use=None)
    assert called == []


def test_all_cells_freeze_before_qeval_and_d1_stays_out_of_baseline_cells(tmp_path):
    plan = registration(four=True)
    plan.update(execution_authorized=True, approval_ref={'file': 'explicit_test_approval.json', 'json_path': []})
    plan['quote'] = runner.quote_method_action_pairing(plan)
    agents, observed, future_labels = {}, [], [99]

    def restore(name, contract, output):
        agent, lab = agent_scene(output)
        lab.public_spec = lambda: {'action_kinds': ['trajectory'], 'controls': ['u'],
            'max_probe_length': 8, 'feature_bounds': {'u': (-1., 1.)}}
        rows = [Observation({'u': 0.}, 1.)]
        if name.endswith('1'):
            rows.append(Observation({'u': 1.}, 2.))
        agent.rows = rows
        def no_deep_result(*args, **kwargs):
            raise AssertionError('checkpoint must use checkpoint_result, never deep result')
        agent.result = no_deep_result
        actual_checkpoint = agent.checkpoint_result
        def light_checkpoint(status):
            assert status == 'researching'
            return actual_checkpoint(status)
        agent.checkpoint_result = light_checkpoint
        agent.config = replace(agent.config, compute_budget=10000, observation_budget=0,
            execution_seconds=30, confirmation_rows=96)
        agent.costs['observation_cost'] = 0
        agent._save('old.py', export_temporal_program(Program('const', value=1.), feature_names=('u',)))
        agent._save('method.py', "def select_actions(public_spec, observations, memory):\n    return {'model':'old.py'}\n")
        question = agent._open_question({'id': 'q', 'question': 'Compare paid synthetic candidates',
            'parent_id': 'task', 'answer_type': 'number', 'context_ref': {'domain': 'synthetic'}})
        agents[name] = agent
        profile = output / 'profile.json'; profile.write_text('{}')
        artifact = output / 'artifact.json'; artifact.write_text(json.dumps(contract['method_ref']))
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': question['id']},
            'profile_ref': {'file': str(profile), 'json_path': []},
            'method_artifact_ref': {'file': str(artifact), 'json_path': []},
            'origin_ref': {'file': str(output / 'invocation.json'), 'json_path': []},
            **{key: contract[key] for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    def evaluate(name, frozen, evaluation_ref, agent):
        manifest = json.loads((tmp_path / 'study/frozen_outputs.json').read_text())
        assert len(manifest['models']) == 4
        assert manifest['evaluation_labels_opened'] == bool(observed)
        source = Path(frozen['workspace']) / frozen['model']
        before = source.read_text()
        future_labels[0] += 1
        assert source.read_text() == before
        observed.append(name)
        assert len(agent.rows) == (2 if name.endswith('1') else 1)
        agent.costs['compute_cost'] += 1
        return {'loss': .1, 'quality': {'loss': .1, 'threshold': 1e-10}, 'coverage': 24, 'completed': False}

    def ordinary(pair_refs, outcome_refs, output):
        assert len(pair_refs) == 2 and len(outcome_refs) == 4
        assert len(observed) == 4
        challenger = json.loads(Path(outcome_refs['10']['file']).read_text())
        assert challenger['method_ref'] == plan['cells']['10']['method_ref']
        canonical = Path(challenger['continuation_state_ref']['file'])
        assert canonical == agents['10'].workspace / 'training.json' and canonical.is_file()
        assert not (tmp_path / 'study/10/state.json').exists()
        assert challenger['continuation_data_ref']['json_path'] == ['paid_observations']
        assert challenger['continuation_scope'] == {'paid_rows': 1, 'pool_models': 1}
        assert challenger['costs_after'] == {**agents['10'].costs,
            'compute_cost': agents['10'].costs['compute_cost'] + 1}
        assert challenger['continuation_costs_ref']['json_path'] == ['costs_after']
        assert Path(challenger['method_artifact_ref']['file']).name == 'artifact.json'
        return {'status': 'same_common_route_received_actual_pair_refs'}

    result = runner.run_method_action_pairing(plan, tmp_path / 'study', restore_cell=restore,
        evaluate_frozen=evaluate, ordinary_use=ordinary)
    assert result['status'] == 'completed_development_pairing'
    assert set(observed) == {'00', '10', '01', '11'}
    assert all(agent.costs['observation_cost'] == 0 for agent in agents.values())
    assert agents['00'].rows is not agents['10'].rows


def test_confirmation_reserve_applies_only_when_confirmation_is_permitted(tmp_path):
    agent, _ = agent_scene(tmp_path)
    agent.config = replace(agent.config, observation_budget=0, confirmation_rows=96)
    agent._procedure_confirmation_allowed = True
    assert agent._procedure_budget_status() == 'confirmation_budget_exhausted'
    agent._procedure_confirmation_allowed = False
    assert agent._procedure_budget_status() is None


def test_old_python_predictor_keeps_its_frozen_execution_dependencies(tmp_path):
    from types import SimpleNamespace

    workspace = tmp_path / 'old'; workspace.mkdir()
    (workspace / 'helper.py').write_text('VALUE = 2\n')
    source = 'from helper import VALUE\ndef predict(row):\n    return VALUE * row["u"]\n'
    (workspace / 'old.py').write_text(source)
    frozen = runner._freeze(SimpleNamespace(workspace=workspace), 'old.py', tmp_path / 'frozen')
    destination = Path(frozen['workspace'])
    assert (destination / frozen['model']).read_text() == source
    assert (destination / 'helper.py').read_text() == 'VALUE = 2\n'


@pytest.mark.parametrize('nested_score', [False, True])
def test_full_comparison_is_inside_same_deadline_and_failed_cells_remain_unmeasured(
        tmp_path, monkeypatch, nested_score):
    import time
    from hegel_machine.cognition.experiment_runtime import PythonExecutionDeadline, python_execution_deadline

    plan = registration()
    for cell in plan['cells'].values():
        cell['time_limits']['active_wall_seconds'] = .01
    plan.update(execution_authorized=True, approval_ref={'file': 'test_approval.json', 'json_path': []})
    plan['quote'] = runner.quote_method_action_pairing(plan)
    monkeypatch.setattr(runner, 'execute_procedure', lambda *args, **kwargs:
        {'status': 'completed', 'model': 'old.py'})

    def restore(name, contract, output):
        agent, lab = agent_scene(output)
        agent.rows = [Observation({'u': 0.}, 1.)]
        def slow_comparison(*args, **kwargs):
            if nested_score:
                try:
                    with python_execution_deadline(1.):
                        time.sleep(.1)
                except PythonExecutionDeadline:
                    pass
            else:
                time.sleep(.1)
            raise AssertionError('registered active deadline did not cover comparison')
        agent._execute = slow_comparison
        profile = output / 'profile.json'; profile.write_text('{}')
        return {'agent': agent, 'lab': lab, 'path': 'old.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': 'q'},
            'profile_ref': {'file': str(profile), 'json_path': []},
            'origin_ref': {'file': str(output / 'origin.json'), 'json_path': []},
            **{key: contract[key] for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    def no_evaluation(*args):
        raise AssertionError('failed method produced no frozen evaluation subject')

    result = runner.run_method_action_pairing(plan, tmp_path / 'study', restore_cell=restore,
        evaluate_frozen=no_evaluation,
        ordinary_use=lambda *args: {'status': 'original_common_route_receives_unresolved_pairs'},
        prepare_evaluation=lambda: pytest.fail('no frozen predictor can consume new evaluation labels'))
    assert result['shared_evaluation_preparation_ref'] is None
    for reference in result['outcome_refs'].values():
        outcome = json.loads(Path(reference['file']).read_text())
        assert outcome['status'] == 'time_exhausted' and not outcome['execution_complete']
        assert outcome['continuation_scope']['pool_models'] == 1
        assert 'loss' not in outcome and outcome['timing']['rescue_preservation_seconds'] >= 0
        phases = outcome['host_phase_timing']['phase_seconds']
        assert phases['final_comparison'] > 0 and phases['exception_rescue'] >= 0
        assert phases['final_save'] == phases['freeze'] == 0


def test_different_component_prices_keep_equal_actual_execution_caps(tmp_path):
    plan = registration()
    plan['cells']['00']['components']['search']['compute'] = 1
    plan['cells']['10']['components']['search']['compute'] = 20
    plan.update(execution_authorized=True, approval_ref={'file': 'test_approval.json', 'json_path': []})
    plan['quote'] = runner.quote_method_action_pairing(plan)
    assert plan['quote']['status'] == 'quoted'
    agents = {}

    def restore(name, contract, output):
        agent, lab = agent_scene(output)
        lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
            'max_probe_length': 2, 'feature_bounds': {'u': (-1., 1.)}}
        agent.rows = [Observation({'u': 0.}, 1.)]
        agent.config = replace(agent.config, compute_budget=1000, execution_seconds=30, observation_budget=0)
        agent._save('old.py', export_temporal_program(Program('const', value=1.), feature_names=('u',)))
        agent._save('method.py', "def select_actions(public_spec, observations, memory):\n    return {'model':'old.py'}\n")
        question = agent._open_question({'id': 'q', 'question': 'Use paid synthetic evidence',
            'parent_id': 'task', 'answer_type': 'number', 'context_ref': {'domain': 'synthetic'}})
        agents[name] = agent
        profile = output / 'profile.json'; profile.write_text('{}')
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': question['id']},
            'profile_ref': {'file': str(profile), 'json_path': []},
            'origin_ref': {'file': str(output / 'origin.json'), 'json_path': []},
            **{key: contract[key] for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    result = runner.run_method_action_pairing(plan, tmp_path / 'study', restore_cell=restore,
        evaluate_frozen=lambda *args: {'loss': 0., 'quality': {'loss': 0., 'threshold': 1e-10},
            'completed': True, 'coverage': 1}, ordinary_use=lambda *args: {'status': 'retained_actual_refs'})
    assert result['status'] == 'completed_development_pairing'
    assert agents['00'].config.compute_budget == agents['10'].config.compute_budget == 100
    assert plan['cells']['00']['components']['search']['compute'] == 1
    assert plan['cells']['10']['components']['search']['compute'] == 20
    assert result['quote']['compute'] == 2 * 100 + 60
    actual_pair = json.loads(Path(result['pair_refs'][0]['file']).read_text())
    # These two synthetic programs only return their old model: registered
    # names differ, but neither actually chose a different research operation.
    assert actual_pair['baseline_action_content'] == actual_pair['challenger_action_content'] == []


@pytest.mark.parametrize('field', ['compute_cap', 'observation_cap', 'time_limits'])
def test_unequal_pair_cap_or_time_is_rejected_before_restore(tmp_path, field):
    plan = registration()
    plan['observation_cap'] = 10
    if field == 'time_limits':
        plan['cells']['10'][field]['billed_python_seconds'] -= 1
    else:
        plan['cells']['10'][field] += 1
    plan.update(execution_authorized=True, approval_ref={'file': 'test_approval.json', 'json_path': []})
    plan['quote'] = runner.quote_method_action_pairing(plan)
    assert '00/10.equal_resource_caps_and_time_limits_required' in plan['quote']['missing']
    called = []
    with pytest.raises(ValueError):
        runner.run_method_action_pairing(plan, tmp_path,
            restore_cell=lambda *args: called.append(True), evaluate_frozen=None, ordinary_use=None)
    assert called == []


def test_frozen_evaluation_uses_light_accounts_after_full_agents_are_released(tmp_path):
    import weakref

    plan = registration()
    plan.update(execution_authorized=True, approval_ref={'file': 'test_approval.json', 'json_path': []})
    plan['quote'] = runner.quote_method_action_pairing(plan)
    weak_agents = []

    def restore(name, contract, output):
        agent, lab = agent_scene(output)
        lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
            'max_probe_length': 2, 'feature_bounds': {'u': (-1., 1.)}}
        agent.rows = [Observation({'u': 0.}, 1.)]
        agent.config = replace(agent.config, compute_budget=1000, execution_seconds=30, observation_budget=0)
        agent._save('old.py', export_temporal_program(Program('const', value=1.), feature_names=('u',)))
        agent._save('method.py', "def select_actions(public_spec, observations, memory):\n    return {'model':'old.py'}\n")
        question = agent._open_question({'id': 'q', 'question': 'Use paid synthetic evidence',
            'parent_id': 'task', 'answer_type': 'number', 'context_ref': {'domain': 'synthetic'}})
        weak_agents.append(weakref.ref(agent))
        profile = output / 'profile.json'; profile.write_text('{}')
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': question['id']},
            'profile_ref': {'file': str(profile), 'json_path': []},
            'origin_ref': {'file': str(output / 'origin.json'), 'json_path': []},
            **{key: contract[key] for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    evaluated, prepared = [], []
    def prepare():
        manifest = json.loads((tmp_path / 'study/frozen_outputs.json').read_text())
        assert len(manifest['models']) == 2 and all(reference() is None for reference in weak_agents)
        prepared.append(True)
        return {'status': 'prepared_shared_engineering_evaluation', 'observation_cost': 0}

    def evaluate(name, frozen, evaluation_ref, account):
        assert prepared == [True]
        assert len(weak_agents) == 2 and all(reference() is None for reference in weak_agents)
        assert isinstance(account.rows, tuple)
        assert not hasattr(account, 'theory_index') and not hasattr(account, 'theory_consolidation')
        assert not hasattr(account, 'knowledge')
        account.costs['compute_cost'] += 1
        evaluated.append(name)
        return {'loss': 0., 'quality': {'loss': 0., 'threshold': 1e-10}, 'completed': True, 'coverage': 1}

    result = runner.run_method_action_pairing(plan, tmp_path / 'study', restore_cell=restore,
        evaluate_frozen=evaluate, ordinary_use=lambda *args: {'status': 'actual_refs_received'},
        prepare_evaluation=prepare)
    assert result['status'] == 'completed_development_pairing' and evaluated == ['00', '10']
    assert result['shared_evaluation_preparation_ref']


def test_native_query_callback_ends_one_registered_unit_and_preserves_ordinary_hooks(tmp_path, monkeypatch):
    agent, lab = agent_scene(tmp_path)
    order = []

    def execute(action, arguments, laboratory):
        assert agent._unit_research_compute_ceiling == 123
        order.append('query')
        proposal = {'action': 'query', 'arguments': {'actions': []}}
        feedback = {'after_observation_results': [{'operator': 'native_condition_research',
            'native_condition_research': {'round_index': 0, 'status': 'candidates_require_evaluation'}}]}
        agent._current_proposal = proposal
        agent.method_results.append({'proposal': proposal, 'feedback': feedback})
        return feedback

    monkeypatch.setattr(agent, '_execute', execute)
    result = runner.execute_registered_method_unit(agent, lab, 'frozen_method.py', max_steps=48,
        before_step=lambda laboratory: order.append('before'),
        after_step=lambda laboratory: order.append('after'),
        checkpoint=lambda laboratory: order.append('checkpoint'), native_attempt_limit=1,
        research_compute_ceiling=123)
    assert result['status'] == 'registered_native_attempt_returned'
    assert order == ['before', 'query', 'after', 'checkpoint']
    assert result['native_boundary_refs'][0]['json_path'][-2:] == ['after_observation_results', 0]
    assert len(result['actual_action_content']) == 1
    assert not hasattr(agent, '_unit_research_compute_ceiling')


def test_common_before_hook_work_is_not_claimed_as_the_method_program_action(tmp_path, monkeypatch):
    agent, lab = agent_scene(tmp_path)
    called = []

    def before(laboratory):
        agent.operator_results.append({'operator': 'native_condition_research',
            'native_condition_research': {'round_index': 0, 'status': 'candidates_require_evaluation'}})

    def procedure_action(*args, **kwargs):
        called.append(True)
        feedback = {'status': 'author_work_request', 'request': {'reason': 'engineering_control'}}
        agent.method_results.append({'proposal': {'stop': feedback}, 'feedback': feedback})
        return feedback

    monkeypatch.setattr(agent, '_execute', procedure_action)
    result = runner.execute_registered_method_unit(agent, lab, 'not_yet_consumed.py',
        before_step=before, native_attempt_limit=1)
    assert result['status'] == 'author_work_request' and called == [True]
    assert result.get('native_boundary_refs', []) == []
    assert result['actual_action_content'] == [] and result['actual_action_refs'] == []


def test_structure_planning_returns_to_the_method_before_the_actual_native_fit(tmp_path, monkeypatch):
    agent, lab = agent_scene(tmp_path)
    calls = []

    def action(kind, arguments, laboratory):
        first = not calls
        calls.append('planning' if first else 'query')
        proposal = ({'action': 'run', 'arguments': {'operator': 'native_condition_research',
                     'structure_planning': {'round_index': 0}}} if first else
                    {'action': 'query', 'arguments': {'actions': []}})
        feedback = ({'operator': 'native_condition_research',
            'structure_planning': {'status': 'structure_experiments_planned'}} if first else
            {'after_observation_results': [{'operator': 'native_condition_research',
                'native_condition_research': {'status': 'candidates_require_evaluation'}}]})
        if first:
            agent.operator_results.append(deepcopy(feedback))
        agent._current_proposal = proposal
        agent.method_results.append({'proposal': proposal, 'feedback': feedback})
        return feedback

    monkeypatch.setattr(agent, '_execute', action)
    result = runner.execute_registered_method_unit(agent, lab, 'method.py',
        max_steps=3, native_attempt_limit=1)
    assert calls == ['planning', 'query']
    assert result['registered_attempt_returned'] is True
    assert len(result['native_boundary_refs']) == 1
    assert result['native_boundary_refs'][0]['json_path'][:2] == ['method_results', 1]
    assert len(result['actual_action_content']) == 2


@pytest.mark.parametrize('failure_kind', ['one_incomplete_cell', 'preparation_only'])
def test_incomplete_method_delivery_does_not_buy_or_open_qeval(tmp_path, monkeypatch, failure_kind):
    plan = registration()
    if failure_kind == 'preparation_only':
        for cell in plan['cells'].values():
            cell.update(planned_native_compute=1,
                        actual_candidate_complete_quote={'bounded_research_compute': 10})
        execute = runner.execute_registered_method_unit
        def with_complete_core(*args, **kwargs):
            assert kwargs['research_compute_reserve'] == 10
            return execute(*args, **kwargs)
        monkeypatch.setattr(runner, 'execute_registered_method_unit', with_complete_core)
    plan.update(execution_authorized=True, approval_ref={'file': 'explicit_fixture_approval'})
    plan['quote'] = runner.quote_method_action_pairing(plan)
    calls = []

    def restore(name, contract, output):
        agent, lab = agent_scene(output)
        lab.public_spec = lambda: {'action_kinds': ['trajectory'], 'controls': ['u'],
            'max_probe_length': 8, 'feature_bounds': {'u': (-1., 1.)}}
        agent.rows = [Observation({'u': 0.}, 1.)]
        agent.config = replace(agent.config, compute_budget=10000, observation_budget=0,
            execution_seconds=30)
        agent._save('old.py', export_temporal_program(Program('const', value=0.), feature_names=('u',)))
        agent._save('method.py', "def select_actions(public_spec, observations, memory):\n    return {'model':'old.py'}\n")
        question = agent._open_question({'id': 'q', 'question': 'fixture fixed parent', 'parent_id': 'task'})
        if name == '10' and failure_kind == 'one_incomplete_cell':
            def interrupted(*_):
                raise RuntimeError('incomplete fixture cell')
            monkeypatch.setattr(agent, '_execute', interrupted)
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': ['old.py'],
            'scope': contract['scope'], 'question_ref': {'id': question['id']},
            'profile_ref': {'file': 'profile.json'}, 'origin_ref': {'file': 'origin.json'},
            **({'native_attempt_limit': 1} if failure_kind == 'preparation_only' else {}),
            **{key: contract[key] for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    def forbidden(*_):
        pytest.fail('incomplete delivery must not acquire or evaluate Qeval')

    result = runner.run_method_action_pairing(plan, tmp_path / 'run', restore_cell=restore,
        prepare_evaluation=forbidden, evaluate_frozen=forbidden,
        ordinary_use=lambda *args: {'status': 'unresolved_without_ordinary_execution'})
    assert result['evaluation_lifecycle']['status'] == 'blocked_incomplete_outputs'
    assert result['evaluation_lifecycle']['labels_revealed'] is False
    assert result['shared_evaluation_preparation_ref'] is None
    if failure_kind == 'preparation_only':
        for name in ('00', '10'):
            outcome = json.loads((tmp_path / 'run' / name / 'outcome.json').read_text())
            assert outcome['comparison']['models_requested'] == outcome['comparison']['models_completed'] == 1
            assert outcome['method_delivery'] == 'preparation_only' and not outcome['execution_complete']


def test_research_primitive_cannot_consume_the_held_complete_comparison_budget(tmp_path, monkeypatch):
    for operator in ('refit', 'native_condition_research'):
        agent, lab = agent_scene(tmp_path / operator)
        agent.config = replace(agent.config, compute_budget=100)
        agent.costs['compute_cost'] = 20
        calls = []

        def action(action, arguments, laboratory):
            calls.append((arguments.get('operator'), agent.config.compute_budget,
                agent._remaining_compute()))
            if arguments.get('operator') == 'native_condition_research':
                assert arguments['max_compute'] == 40
            agent._charge(40)
            return {'status': 'bounded_fixture_work'}

        def procedure(actual_agent, laboratory, path, **kwargs):
            assert actual_agent.config.compute_budget == 100  # ordinary selection sees the whole funded unit
            actual_agent._execute('run', {'operator': operator}, laboratory)
            return {'status': 'completed'}

        monkeypatch.setattr(agent, '_execute_action', action)
        monkeypatch.setattr(runner, 'execute_procedure', procedure)
        runner.execute_registered_method_unit(agent, lab, 'method.py', research_compute_ceiling=60)
        assert agent.config.compute_budget == 100 and not hasattr(agent, '_unit_research_compute_ceiling')
        assert not hasattr(agent, '_unit_full_compute_budget')
        agent._execute('run', {'operator': 'research_choice'}, lab)
        expected = (operator, 100, 80) if operator == 'native_condition_research' else (operator, 60, 40)
        assert calls == [expected, ('research_choice', 100, 40)]
        assert agent.costs['compute_cost'] == 100


def test_real_native_budget_holds_full_pool_once_and_keeps_the_quoted_joint_attempt(tmp_path, monkeypatch):
    from test_native_complete_research_unit import paid_scene
    from hegel_machine.cognition import structural_inquiry as native

    agent, question, previous, arguments = paid_scene(tmp_path)
    agent.config = replace(agent.config, compute_budget=2000000, confirmation_rows=96)
    agent.knowledge['finite_native_author']['predictor_paths'] = [
        f'archive/paid_{index:04d}.py' for index in range(4762)]
    agent.rows.extend(Observation({'u': 1.}, 0.) for _ in range(24))
    agent.observations.append({'row_indices': list(range(145, 169))})
    previous['candidates'] = [{'program': Program('const', value=float(index)).to_dict()}
        for index in range(182)]
    question['native_condition_research'] = [previous]
    agent.evaluated_model = {'model': 'old.py', 'evaluated_rows': 145}
    reserve = native._native_completion_reserve(agent, question, 246)
    assert reserve['comparison_compute_cost'] == 846521 and reserve['compute_cost'] == 846786
    arguments = {**arguments, 'question': {'id': question['id']}, 'max_compute': 520819}

    class BackendReached(RuntimeError):
        pass

    def inspect_budget(public_spec, observations, budget, **options):
        assert len(agent.rows) == 169 and len(options['retained_candidates']) == 182
        assert agent.config.compute_budget == 2000000
        assert budget.limit == 520819
        assert agent.config.compute_budget - agent.costs['compute_cost'] - budget.limit >= reserve['compute_cost'] + 24
        raise BackendReached('budget construction only; no native fitting')

    def procedure(actual_agent, laboratory, path, **kwargs):
        return actual_agent._execute('run', arguments, laboratory)

    monkeypatch.setattr(native, 'research_native_conditions', inspect_budget)
    monkeypatch.setattr(runner, 'execute_procedure', procedure)
    laboratory = type('PublicOnlyLab', (), {'public_spec': lambda self: {'controls': ['u']}})()
    with pytest.raises(BackendReached, match='no native fitting'):
        runner.execute_registered_method_unit(agent, laboratory, 'method.py',
            research_compute_ceiling=1153190)
    assert agent.config.compute_budget == 2000000 and agent.costs['compute_cost'] == 0
    assert not hasattr(agent, '_unit_full_compute_budget')


def test_command_line_requires_named_inputs_and_refuses_unapproved_science_before_restore(tmp_path, monkeypatch):
    from scripts import prepare_pr3_source_successor

    constructed = []
    def callbacks(registration, output):
        constructed.append(True)
        def no_restore(*args):
            pytest.fail('unapproved CLI input reached scientific restore')
        return {'restore_cell': no_restore, 'evaluate_frozen': None, 'ordinary_use': None}
    monkeypatch.setattr(prepare_pr3_source_successor, 'make_retained_method_pairing_factories', callbacks)
    with pytest.raises(SystemExit) as missing:
        runner.main(['--registration', str(tmp_path / 'plan.json')])
    assert missing.value.code == 2 and not constructed
    path = tmp_path / 'plan.json'; path.write_text(json.dumps(registration()))
    with pytest.raises(ValueError, match='explicit approval'):
        runner.main(['--registration', str(path), '--output', str(tmp_path / 'science')])
    assert constructed == [True] and not (tmp_path / 'science').exists()


def test_ordinary_use_execution_caps_cannot_exceed_the_quoted_global_reservation(tmp_path):
    plan = registration()
    plan['ordinary_use'] = {'compute_cap': 10, 'observation_cap': 0}
    assert runner.quote_method_action_pairing(plan)['status'] == 'quoted'
    for field, value in [('compute_cap', 11), ('observation_cap', 1)]:
        bad = deepcopy(plan); bad['ordinary_use'][field] = value
        quote = runner.quote_method_action_pairing(bad)
        assert quote['status'] == 'quote_incomplete'
        assert f'ordinary_use.{field}.exceeds_registered_reservation' in quote['missing']
        bad.update(execution_authorized=True, approval_ref={'file': 'test_approval.json'}, quote=quote)
        with pytest.raises(ValueError, match='current complete frozen quote'):
            runner.run_method_action_pairing(bad, tmp_path / field,
                restore_cell=lambda *args: pytest.fail('an over-reserved ordinary use reached restore'),
                evaluate_frozen=None, ordinary_use=None)


def test_failed_unit_preserves_partial_hook_checkpoint_and_rescue_phases_without_new_fees(tmp_path, monkeypatch):
    from hegel_machine.cognition import experiment_runtime

    clock = [0.]
    monkeypatch.setattr(experiment_runtime, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(experiment_runtime, 'process_time', lambda: clock[0])
    monkeypatch.setattr(runner, 'monotonic', lambda: clock[0])
    plan = registration()
    plan.update(execution_authorized=True, approval_ref={'file': 'test_approval.json'})
    plan['quote'] = runner.quote_method_action_pairing(plan)

    def advance(seconds):
        clock[0] += seconds

    def restore(name, contract, output):
        agent, lab = agent_scene(output)
        agent.rows = [Observation({'u': 0.}, 1.)]
        monkeypatch.setattr(agent, '_prepare_program_state', lambda: advance(11))
        profile = output / 'profile.json'; profile.write_text('{}')
        return {'agent': agent, 'lab': lab, 'path': 'method.py', 'models': [],
            'before_step': lambda laboratory: advance(2),
            'after_step': lambda laboratory: advance(3), 'scope': contract['scope'],
            'question_ref': {'id': 'q'}, 'profile_ref': {'file': str(profile), 'json_path': []},
            'origin_ref': {'file': str(output / 'origin.json'), 'json_path': []},
            **{key: contract[key] for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    def procedure(agent, lab, path, **hooks):
        hooks['before_step'](lab)
        advance(7)
        hooks['after_step'](lab)
        raise RuntimeError('retained procedure failure')

    monkeypatch.setattr(runner, 'execute_procedure', procedure)
    result = runner.run_method_action_pairing(plan, tmp_path / 'study', restore_cell=restore,
        evaluate_frozen=lambda *args: pytest.fail('failed unit has no frozen evaluator'),
        ordinary_use=lambda *args: {'status': 'unresolved_originals_retained'})
    for reference in result['outcome_refs'].values():
        outcome = json.loads(Path(reference['file']).read_text())
        assert outcome['status'] == 'execution_error' and not outcome['execution_complete']
        phases = outcome['host_phase_timing']['phase_seconds']
        assert phases == {'before_turn_hooks': 2., 'procedure_including_hooks_and_checkpoint': 23.,
            'after_turn_hooks': 3., 'checkpoint': 22., 'final_comparison': 0., 'freeze': 0.,
            'final_save': 0., 'exception_rescue': 11.}
        assert outcome['timing']['activity_wall_seconds'] == 23.
        assert outcome['timing']['rescue_preservation_seconds'] == 11.
        assert outcome['host_phase_timing']['wall_seconds'] == 34.
        assert sum(phases.values()) > outcome['host_phase_timing']['wall_seconds']
        assert outcome['host_phase_timing']['historical_phase_times_reconstructed'] is False
        assert outcome['costs']['compute_cost'] == outcome['costs']['observation_cost'] == 0
