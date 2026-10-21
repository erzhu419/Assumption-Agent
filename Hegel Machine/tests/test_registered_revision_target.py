"""Actual Registered hooks bind retained Programs and resolve paid rows by reference."""
from copy import deepcopy
from dataclasses import replace
import json
from math import sqrt

import pytest

from hegel_machine.cognition import revision_relations
from hegel_machine.cognition.question_routing import register_theories, resolved_theories, scoped_numeric_refutation
from hegel_machine.cognition.theory_consolidation import _checked_transfer, _resolve_paid_checks
from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent, ORGANIZATION_ORIGIN
from test_finite_native_author import ChoiceLab
from test_inquiry_revision import action
from test_native_condition_inquiry import PaidConditionalLab
from test_question_routing import CONTEXT, ROLES, theory
from test_registered_finite_policy import registered_scene
from test_revision_context_fitting import FITTING
from test_revision_relations import edit_candidate


def revision_scene(tmp_path):
    agent, lab, declaration = registered_scene(tmp_path)
    agent.config = replace(agent.config, compute_budget=20000)
    owner = next(q for q in agent.questions if q['id'] == declaration['id'])
    owner['theory_target']['revision_fitting'] = {
        **FITTING, 'operations': ['refit'], 'max_structures': 1}
    frozen = {**edit_candidate(), 'version': 1}
    library = tmp_path / 'frozen' / 'shared' / 'theory_library.json'
    library.parent.mkdir(parents=True)
    library.write_text(json.dumps({'candidates': {frozen['id']: frozen}}))
    agent._organization_state['automatic_library_path'] = str(library)
    agent._organization_state['initial_library_path'] = str(library)
    agent._organization_registration['initial_library_path'] = str(library)
    agent._active_question_id = owner['id']
    return agent, lab, owner


def forbid_acquisition(lab, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Revision consumption must use the retained fixture prefix')
    monkeypatch.setattr(lab, 'execute', forbidden)


def test_registered_hook_uses_actual_retained_native_parent_and_live_paid_rows_then_restores(tmp_path, monkeypatch):
    agent, lab, owner = revision_scene(tmp_path)
    # Select by a real evaluation, independently of the bank's own ranking.
    bank = owner['native_condition_research'][0]['candidates']
    candidate = bank[-1]
    evaluation = agent._execute('evaluate', {'model': candidate['model']}, lab)
    assert evaluation['retention_decision']['status'] == 'candidate_retained'
    selected = agent.evaluated_model['source_model']
    assert selected == candidate['model']
    # A request file can predate a newly purchased record. Resolution must use
    # agent.observations, never silently use this earlier disk payload.
    old_observations = (agent.workspace / 'laboratory_observations.json').read_text()
    agent._execute('query', {'actions': [action(4.)]}, lab)
    (agent.workspace / 'laboratory_observations.json').write_text(old_observations)
    agent._active_question_id = owner['id']
    before = deepcopy(agent.costs)
    forbid_acquisition(lab, monkeypatch)
    calls = []
    propose = revision_relations.propose_revision

    def observed(candidate, parent, components, **kwargs):
        result = propose(candidate, parent, components, **kwargs)
        calls.append((deepcopy(parent), deepcopy(kwargs['evidence_checks']), deepcopy(result)))
        return result

    monkeypatch.setattr(revision_relations, 'propose_revision', observed)
    agent._before_model_turn(lab)
    assert len(calls) == 1, agent.events[-1]['feedback']
    parent, checks, proposed = calls[0]
    target = owner['theory_target']
    assert parent['id'] == target['parent_theory_id']
    assert agent.theory_index[parent['id']]['source_ref']['candidate_index'] == len(bank) - 1
    assert parent['program'] == candidate['program']
    assert len(checks) == len(agent.rows) == 2
    assert [check['observation']['target'] for check in checks] == [1., 4.]
    assert all(check['tolerance'] == sqrt(2 * agent.config.confirmation_mse_threshold) for check in checks)
    assert proposed['proposals']
    assert any(item['fitting']['fitted_rows'] == 2 for item in proposed['proposals'])
    assert all('observation' not in check for check in target['evidence_checks'])
    executions = [item for key, item in agent.theory_consolidation['executions'].items()
                  if key.startswith('relation_transfer:')]
    assert len(executions) == 1
    assert all('observation' not in check for check in executions[0]['request_basis']['evidence_checks'])
    assert agent.evaluated_model['source_model'] == selected
    assert 0 < agent.costs['compute_cost'] - before['compute_cost'] <= 20000 - before['compute_cost']
    assert agent.costs['observation_cost'] == before['observation_cost'] and agent.costs['model_calls'] == 0
    assert agent.knowledge['finite_native_author']['predictor_paths'] == ['archive/failed.py']
    event = next(item for item in reversed(agent.events) if item.get('origin') == ORGANIZATION_ORIGIN)
    assert event['feedback']['frozen_recall']['work']
    assert not any(identifier.startswith('native_component:') for identifier in agent.theory_index)
    repeated_costs = deepcopy(agent.costs)
    agent._active_question_id = owner['id']
    agent._before_model_turn(lab)
    assert len(calls) == 1 and agent.costs == repeated_costs
    # Restore the ordinary Common checkpoint and resolve the same tiny refs.
    agent._write_evidence()
    agent._prepare_program_state()
    previous = agent.result('researching')
    restored = RegisteredOrganizationAgent(None, tmp_path / 'restored', arm='fixed_pr2',
        research_organization=deepcopy(agent._organization_registration), source_task=False,
        knowledge={'files': {}}, config=agent.config)
    class RestorableChoiceLab(ChoiceLab):
        costs = PaidConditionalLab.costs
        restore_paid_account = PaidConditionalLab.restore_paid_account

    restored_lab = RestorableChoiceLab()
    forbid_acquisition(restored_lab, monkeypatch)
    restored.restore_research(previous, agent.workspace, restored_lab, confirmation_seed=0)
    recovered = next(q for q in restored.questions if q['id'] == owner['id'])['theory_target']
    assert recovered['evidence_checks'] == target['evidence_checks']
    assert [check['observation']['target'] for check in _resolve_paid_checks(
        restored, recovered['evidence_checks'])] == [1., 4.]
    assert restored.costs == agent.costs


def test_registered_hook_keeps_opaque_parent_obligation_when_native_paid_scope_is_stale(tmp_path, monkeypatch):
    agent, lab, owner = revision_scene(tmp_path)
    agent._execute('evaluate', {'model': 'archive/failed.py'}, lab)
    # The old failure is real, but does not assess this newly acquired row.
    agent._execute('query', {'actions': [action(4.)]}, lab)
    before = deepcopy(agent.costs)
    forbid_acquisition(lab, monkeypatch)

    def no_revision(*args, **kwargs):
        raise AssertionError('An opaque retained Python model is not a native parent Program')
    monkeypatch.setattr(revision_relations, 'propose_revision', no_revision)
    agent._active_question_id = owner['id']
    agent._before_model_turn(lab)
    target = owner['theory_target']
    assert 'parent_theory_id' not in target and 'evidence_checks' not in target
    assert target['parent_binding_obligation']['reason'] == 'unique_retained_native_parent_required'
    event = next(item for item in reversed(agent.events) if item.get('origin') == ORGANIZATION_ORIGIN)
    assert 'frozen_recall' in event['feedback'], event['feedback']
    requests = event['feedback']['frozen_recall']['author_work']
    assert any(item['reason'] == 'unique_retained_native_parent_required' for item in requests)
    assert agent.evaluated_model['source_model'] == 'archive/failed.py'
    assert agent.costs['observation_cost'] == before['observation_cost'] and agent.costs['model_calls'] == 0


def test_paid_mse_rejects_collective_failure_even_when_no_single_residual_is_sufficient(tmp_path):
    agent, lab, owner = revision_scene(tmp_path)
    agent.config = replace(agent.config, confirmation_mse_threshold=1.)
    # PublicLab's real fixture rows have y=u; constant prediction below has
    # e²=1.5 for both rows, below N*tau=2 but with aggregate MSE>tau.
    agent._execute('query', {'actions': [action(1.)]}, lab)
    from hegel_machine.cognition.language import Program
    program = Program('const', value=1. + sqrt(1.5))
    owner['theory_target']['parent_theory_id'] = 'explicit_fixture_parent'
    agent._bind_revision_target(owner)
    checks = deepcopy(owner['theory_target']['evidence_checks'])
    arguments = {'theory_id': 'aggregate_checked', 'context_ref': deepcopy(CONTEXT),
        'role_signature': deepcopy(ROLES), 'output_unit': 'V', 'evidence_checks': checks}
    binding = {'candidate_id': 'aggregate_fixture', 'mapping': {}, 'parameters': [],
               'context_ref': deepcopy(CONTEXT)}
    candidate = {'id': 'aggregate_fixture', 'parameter_paths': []}
    before = agent.costs['compute_cost']
    partial = _checked_transfer(agent, arguments, candidate, program, binding, compute_allowance=1)
    assert partial['status'] == 'UNKNOWN' and partial['paid_mse_check']['status'] == 'UNKNOWN'
    complete = _checked_transfer(agent, arguments, candidate, program, binding, compute_allowance=1)
    assert complete['new_compute_cost'] == 1
    assert all(check['status'] == 'passed' for check in complete['checks'])
    assert complete['status'] == 'refuted'
    assert complete['paid_mse_check']['mse'] == pytest.approx(1.5)
    assert complete['paid_mse_check']['status'] == 'failed'
    assert agent.costs['compute_cost'] - before == 2
    assert agent.theory_index['aggregate_checked']['compatible'] is False
    repeated = _checked_transfer(agent, arguments, candidate, program, binding, compute_allowance=20)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0
    assert repeated['paid_mse_check'] == complete['paid_mse_check']
    expected_ref = {'file': 'shared/theory_library.json', 'json_path': ['transfers',
        len(agent.theory_consolidation['transfers']) - 1, 'paid_mse_check']}
    register_theories(agent, [theory('ordinary_aggregate_alias', program)])
    alias = next(item for item in resolved_theories(agent) if item['id'] == 'ordinary_aggregate_alias')
    assert alias['compatible'] is False
    assert expected_ref in alias['scope_refutation']['counterexample_refs']
    pending = scoped_numeric_refutation(agent, theory('unregistered_aggregate_alias', program))
    assert expected_ref in pending['counterexample_refs']
    assert agent.costs['compute_cost'] - before == 2
