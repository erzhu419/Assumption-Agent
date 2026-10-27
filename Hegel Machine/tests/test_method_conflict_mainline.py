"""Mixed actual method uses become diagnostics and an executable policy."""

from copy import deepcopy

from hegel_machine.cognition.method_feedback import _read, procedure_use_decision
from test_common_method_feedback import scene, use_and_compare, transfer_arguments
from test_method_use_question_route import publish


def mixed_scene(tmp_path):
    references = []
    for index, useful in enumerate((True, True, False)):
        agent, lab, method, parent = scene(tmp_path / str(index), useful=useful)
        use_and_compare(agent, lab, parent)
        reference, = agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs']
        references.append(reference)
    agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs'] = references
    execution = next(value for key, value in agent.theory_consolidation['executions'].items()
                     if key.startswith('research_program_transfer:'))
    return agent, lab, method, parent, deepcopy(execution['basis']['binding'])


def test_mixed_returns_execute_paid_diagnostics_and_change_ordinary_method_call(tmp_path, monkeypatch):
    agent, lab, method, parent, binding = mixed_scene(tmp_path)
    question = publish(agent, method, parent, binding)
    original = deepcopy(method)
    pool = set(agent.theory_consolidation['candidates'])
    result = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}, 'allocation': {'compute': 2000}}, lab)
    research = result['method_conflict_research']
    assert research['condition'] and research['diagnostics']
    assert research['new_compute_cost'] > 0
    current = agent.theory_consolidation['candidates'][method['id']]
    assert set(agent.theory_consolidation['candidates']) == pool
    assert current['research_program_contract'] == original['research_program_contract']
    assert current['version'] > original['version']
    previous = _read(agent, current['previous_version_ref'])
    assert previous['method_use_conditions'].get('research_ref') is None
    assert _read(agent, current['method_use_conditions']['research_ref'])['diagnostics']

    # Another public binding is not an exact same-evidence deferral. The learned
    # condition must actually be consumed, while old methods remain available.
    parent['theory_target']['context_ref'] = {'apparatus': 'another_public_binding'}
    changed_binding = {**binding, 'context_ref': deepcopy(parent['theory_target']['context_ref'])}
    before = procedure_use_decision(agent, previous, changed_binding)
    after = procedure_use_decision(agent, current, changed_binding)
    assert before['admission'] == 'bounded_exploration'
    assert after['admission'] == 'defer'
    assert after['next_action'] == 'ordinary_research'
    costs = deepcopy(agent.costs)

    def forbidden(*args, **kwargs):
        raise AssertionError('the learned negative branch must not fit the method')

    monkeypatch.setattr('hegel_machine.cognition.research_programs.replay_research_program', forbidden)
    feedback = agent._execute('run', transfer_arguments(current, parent, 'conditioned_visit'), lab)
    assert feedback['status'] == 'method_use_deferred'
    assert agent.costs == costs and lab.paid == 0
    assert not feedback['support_updated']
    again = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}, 'allocation': {'compute': 2000}}, lab)
    assert again['new_compute_cost'] == 0


def test_process_return_reference_updates_original_origin_without_another_method_fit(tmp_path):
    agent, lab, method, parent, binding = mixed_scene(tmp_path)
    reference = agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs'][-1]
    comparison = {'method_ref': {'id': method['id'], 'version': method['version']},
        'scope': 'matched_target_configuration_return_not_single_method_causality_or_source_net',
        'same_task': True, 'same_evaluation_contract': True,
        'source_refs': [{'file': 'retained_paired_effect.json', 'json_path': []}],
        'before': {'execution_complete': True, 'completed': True,
            'quality': {'loss': 1e-21, 'threshold': 1e-10}, 'coverage': 512,
            'costs': {'observation_cost': 192, 'compute_cost': 165880}},
        'after': {'execution_complete': True, 'completed': True,
            'quality': {'loss': 2e-21, 'threshold': 1e-10}, 'coverage': 512,
            'costs': {'observation_cost': 168, 'compute_cost': 101218}}}
    path = agent.workspace / 'retained_process_result.json'
    from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
    write_json_checkpoint(path, comparison)
    args = {'operator': 'theory_consolidation', 'object_type': 'method_use_condition',
        'candidate_ids': [method['id']], 'method_process_comparison_refs': [
            {'feedback_ref': reference, 'comparison_ref': {'file': str(path), 'json_path': []}}],
        'mission': {'purpose': 'method_learning', 'allocation': {'compute': 20}},
        'allocation': {'compute': 20}}
    before_cost = agent.costs['compute_cost']
    updated = agent._execute('run', args, lab)
    record = _read(agent, reference)
    assert _read(agent, record['process_comparison_ref']) == comparison
    assert record['working_comparison']['after_loss'] > record['working_comparison']['before_loss']
    current = agent.theory_consolidation['candidates'][method['id']]
    decision = procedure_use_decision(agent, current, binding)
    assert decision['admission'] == 'admit'
    assert agent.costs['compute_cost'] - before_cost == updated['new_compute_cost']
    paid = deepcopy(agent.costs)
    repeated = agent._execute('run', args, lab)
    assert repeated['new_compute_cost'] == 0 and agent.costs == paid
    group = agent.theory_consolidation['method_feedback'][method['id']]
    assert len(group['outcome_refs']) == 3
    assert lab.paid == 0 and not agent.confirmations


def test_restored_use_updates_locally_once_and_cannot_be_overwritten_by_the_next_use(tmp_path):
    from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
    from test_common_method_future_returns import acquire_new_rows

    agent, lab, method, parent = scene(tmp_path / 'live', useful=True)
    use_and_compare(agent, lab, parent)
    group = agent.theory_consolidation['method_feedback'][method['id']]
    original_ref, = group['outcome_refs']
    original = _read(agent, original_ref)
    source = tmp_path / 'retained_readonly_use.json'
    write_json_checkpoint(source, original)
    inherited = {'file': str(source), 'json_path': []}
    group['outcome_refs'] = [inherited]
    group['returns']['origins'][0]['feedback_ref'] = deepcopy(inherited)
    preserved = source.read_bytes()
    result = {'method_ref': original['method_ref'], 'scope': 'matched_target_configuration',
        'same_task': True, 'same_evaluation_contract': True, 'source_refs': [inherited],
        'before': {'execution_complete': True, 'completed': True,
            'quality': {'loss': 1e-21, 'threshold': 1e-10}, 'coverage': 512,
            'costs': {'observation_cost': 192, 'compute_cost': 165880}},
        'after': {'execution_complete': True, 'completed': True,
            'quality': {'loss': 2e-21, 'threshold': 1e-10}, 'coverage': 512,
            'costs': {'observation_cost': 168, 'compute_cost': 101218}}}
    process = tmp_path / 'retained_process.json'
    write_json_checkpoint(process, result)
    args = {'operator': 'theory_consolidation', 'object_type': 'method_use_condition',
        'candidate_ids': [method['id']], 'method_process_comparison_refs': [
            {'feedback_ref': inherited, 'comparison_ref': {'file': str(process), 'json_path': []}}],
        'mission': {'purpose': 'method_learning', 'allocation': {'compute': 20}},
        'allocation': {'compute': 20}}
    agent._execute('run', args, lab)
    local, = group['outcome_refs']
    assert local != inherited and _read(agent, local)['origin_ref'] == inherited
    assert source.read_bytes() == preserved
    costs = deepcopy(agent.costs)
    repeated = agent._execute('run', args, lab)
    assert repeated['new_compute_cost'] == 0 and agent.costs == costs
    assert len(group['returns']['origins']) == 1
    assert group['returns']['origins'][0]['feedback_ref'] == inherited
    assert group['returns']['origins'][0]['fit_compute_cost'] == original['fit_compute_cost']

    measured = acquire_new_rows(agent, lab, parent)
    use_and_compare(agent, measured, parent)
    assert len(group['outcome_refs']) == 2
    assert len({ref['file'] for ref in group['outcome_refs']}) == 2
    assert _read(agent, local)['evidence_basis']['paid_cutoff']['rows'] == 16
    assert source.read_bytes() == preserved
    assert len(group['returns']['origins']) == 2


def test_conflict_quote_includes_new_preparation_and_reuses_already_paid_preparation(tmp_path):
    agent, lab, method, parent, binding = mixed_scene(tmp_path)
    question = publish(agent, method, parent, binding)
    args = {'operator': 'research_route', 'research_question': {'id': question['id']},
        'allocation': {'compute': 3}}
    initial = agent._execute('run', args, lab)
    diagnostic = initial['method_conflict_research']
    assert initial['status'] == 'scoped_unknown'
    assert initial['new_compute_cost'] == 3
    assert diagnostic['new_compute_cost'] == 0
    assert initial['required_compute'] == 3 + diagnostic['required_compute']
    assert not agent.theory_consolidation['candidates'][method['id']][
        'method_use_conditions'].get('research_ref')
    paid = agent.costs['compute_cost']
    args['allocation']['compute'] = diagnostic['required_compute']
    completed = agent._execute('run', args, lab)
    assert completed['status'] == 'method_use_decided'
    assert completed['condition_learning']['new_compute_cost'] == 0
    assert 0 < agent.costs['compute_cost'] - paid <= args['allocation']['compute']
    current = deepcopy(agent.theory_consolidation['candidates'][method['id']])
    cached = agent._execute('run', {'operator': 'theory_consolidation',
        'object_type': 'method_use_condition', 'candidate_ids': [method['id']],
        'mission': question['mission'], 'allocation': {'compute': 0}}, lab)
    assert cached['frozen_candidates'][0]['cache_hit']
    assert cached['new_compute_cost'] == 0
    assert agent.theory_consolidation['candidates'][method['id']] == current


def test_unresolved_new_public_state_proposes_an_actual_bounded_transfer(tmp_path):
    from test_common_method_future_returns import acquire_new_rows

    agent, lab, method, parent, binding = mixed_scene(tmp_path)
    question = publish(agent, method, parent, binding)
    args = {'operator': 'research_route', 'research_question': {'id': question['id']},
        'allocation': {'compute': 2000}}
    agent._execute('run', args, lab)
    measured = acquire_new_rows(agent, lab, parent)
    result = agent._execute('run', args, measured)
    assert result['method_use_decision']['admission'] == 'bounded_exploration'
    assert result['selected_research_action'] == 'bounded_method_trial'
    proposal = result['next_action']
    assert proposal['action'] == 'run'
    assert proposal['arguments']['operator'] == 'theory_transfer'
    assert proposal['arguments']['candidate_id'] == method['id']
    assert proposal['arguments']['research_question']['id'] == parent['id']
    assert 0 < proposal['arguments']['allocation']['compute'] <= 2000
    assert result['method_conflict_research']['next_diagnostic']['proposal'] == proposal
    assert not result['support_updated'] and not result['independent_confirmation']
