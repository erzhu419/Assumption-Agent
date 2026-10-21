"""Partial retained checks resume within the same construction/validation budget."""

from copy import deepcopy
from dataclasses import replace

from hegel_machine.cognition import revision_relations
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.theory_consolidation import _checked_transfer, _save
from test_question_routing import CONTEXT, ROLES, theory
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, constant, edit_candidate, feature
from test_theory_consolidation import agent_scene, assert_no_new_science


def test_zero_budget_checks_resume_only_unknown_rows_and_keep_an_executed_failure(tmp_path, monkeypatch):
    agent, lab = agent_scene(tmp_path)
    program = feature()
    candidate = {'id': 'retained_fixture', 'parameter_paths': []}
    binding = {'candidate_id': candidate['id'], 'mapping': {'role_0': 'u'},
               'parameters': [], 'context_ref': deepcopy(CONTEXT)}
    arguments = {'theory_id': 'checked_fixture', 'context_ref': deepcopy(CONTEXT),
        'role_signature': deepcopy(ROLES), 'output_unit': 'V',
        'evidence_checks': [{'source_ref': {'file': 'paid_fixture.json', 'json_path': [index]},
            'context_ref': deepcopy(CONTEXT), 'tolerance': 0.,
            'observation': {'features': {'u': value}, 'target': target, 'history': []}}
            for index, (value, target) in enumerate(((1., 1.), (2., 2.), (3., 99.), (4., 4.)))]}
    calls = []
    predict = Program.predict

    def counted(value, observation):
        calls.append(observation.features['u'])
        return predict(value, observation)

    monkeypatch.setattr(Program, 'predict', counted)

    def check(allowance):
        return _checked_transfer(agent, arguments, candidate, program, binding,
                                 compute_allowance=allowance)

    initial = check(0)
    assert initial['status'] == 'UNKNOWN' and initial['new_compute_cost'] == 0
    assert [item['status'] for item in initial['checks']] == ['UNKNOWN'] * 4
    unfunded = check(0)
    assert unfunded['cache_hit'] and unfunded['new_compute_cost'] == 0 and calls == []
    first = check(1)
    assert first['new_compute_cost'] == 1 and calls == [1.]
    assert [item['status'] for item in first['checks']] == ['passed', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN']
    retained_pass = deepcopy(first['checks'][0])
    unfunded = check(0)
    assert unfunded['cache_hit'] and unfunded['checks'][0] == retained_pass
    second = check(1)
    assert second['new_compute_cost'] == 1 and calls == [1., 2.]
    assert second['status'] == 'UNKNOWN' and second['checks'][0] == retained_pass
    assert [item['status'] for item in second['checks']] == ['passed', 'passed', 'UNKNOWN', 'UNKNOWN']
    retained_checks = deepcopy(second['checks'][:2])
    third = check(1)
    assert third['new_compute_cost'] == 1 and calls == [1., 2., 3.]
    assert third['status'] == 'refuted'
    assert third['checks'][:2] == retained_checks
    assert [item['status'] for item in third['checks']] == ['passed', 'passed', 'failed', 'UNKNOWN']
    repeated = check(10)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0 and calls == [1., 2., 3.]
    assert repeated['checks'] == third['checks'] and repeated['status'] == 'refuted'
    assert agent.costs['compute_cost'] == 3
    assert agent.theory_index[arguments['theory_id']]['compatible'] is False
    assert_no_new_science(agent, lab)


def test_common_revision_transfer_reserves_one_full_check_set_inside_its_original_25000(tmp_path, monkeypatch):
    agent, lab = agent_scene(tmp_path)
    agent.config = replace(agent.config, compute_budget=25000)
    candidate = {**edit_candidate(), 'version': 1}
    agent.theory_consolidation = {'candidates': {candidate['id']: candidate}}
    _save(agent)
    parent = binary('mul', constant(5.), feature())
    register_theories(agent, [theory('parent', parent)])
    samples = examples(parent)
    checks = [{**deepcopy(samples[index % len(samples)]), 'context_ref': deepcopy(CONTEXT),
        'source_ref': {'file': 'paid_fixture.json', 'json_path': [index]}} for index in range(97)]
    arguments = {'operator': 'theory_transfer', 'candidate_id': candidate['id'],
        'theory_id': 'checked_revision', 'parent_theory_id': 'parent',
        'context_ref': deepcopy(CONTEXT), 'output_unit': 'V',
        'allocation': {'compute': 25000}, 'evidence_checks': checks,
        'mission': {'purpose': 'task', 'mission_id': 'same_total_checked_budget'},
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}
    calls = []
    propose = revision_relations.propose_revision

    def counted(*args, **kwargs):
        result = propose(*args, **kwargs)
        calls.append((deepcopy(kwargs['allocation']), result['actual_compute_cost']))
        return result

    monkeypatch.setattr(revision_relations, 'propose_revision', counted)
    before = agent.costs['compute_cost']
    result = agent._execute('run', deepcopy(arguments), lab)
    assert len(calls) == 1 and calls[0][0]['compute'] == 25000 - len(checks)
    assert calls[0][1] <= 25000 - len(checks)
    assert result['generated_theories']
    assert any(len(item['checks']) == len(checks)
               and all(check['status'] in {'passed', 'failed'} for check in item['checks'])
               for item in result['generated_theories'])
    checked = sum(check['status'] in {'passed', 'failed'}
                  for item in result['generated_theories'] for check in item['checks'])
    assert result['new_compute_cost'] == calls[0][1] + checked
    assert agent.costs['compute_cost'] - before == result['new_compute_cost'] <= 25000
    execution = agent.theory_consolidation['executions']['relation_transfer:checked_revision']
    assert execution['request_basis']['allocation'] == arguments['allocation'] == {'compute': 25000}
    assert agent.theory_index['parent']['program'] == parent.to_dict()
    assert not result['support_updated'] and not result['independent_confirmation']
    assert_no_new_science(agent, lab)
