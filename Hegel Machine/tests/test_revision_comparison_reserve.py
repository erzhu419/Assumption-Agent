"""Revision search holds, rather than spends, each exported model's scoring cost."""

from copy import deepcopy
from dataclasses import replace

from hegel_machine.cognition import revision_relations
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.theory_consolidation import _save, recall_frozen_theories
from test_question_routing import CONTEXT, ROLES, theory
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, component, constant, edit_candidate, feature
from test_theory_consolidation import agent_scene, assert_no_new_science


def test_whole_fit_batches_leave_real_proposals_comparison_funded(monkeypatch):
    parent = binary('mul', constant(5.), feature())
    checks = examples(parent)
    batches = []
    fit = revision_relations._fit_revision_binding

    def observe(*args, **kwargs):
        result = fit(*args, **kwargs)
        batches.append((args[-1].limit, result[2]['actual_fitted_rows'], len(result[0])))
        return result

    monkeypatch.setattr(revision_relations, '_fit_revision_binding', observe)
    result = revision_relations.propose_revision(edit_candidate(), component('parent', parent), [],
        allocation={'compute': 300, 'comparison_rows': 4}, evidence_checks=checks,
        fitting={**FITTING, 'max_structures': 4})
    assert result['proposals'] and batches
    assert all(count <= quota // eligible + 1 for quota, eligible, count in batches if eligible)
    reserved = 4 * len(result['proposals'])
    assert result['reserved_comparison_compute'] == reserved > 0
    assert result['actual_compute_cost'] + reserved <= 300
    assert all(item['parent_binding']['component_id'] == 'parent' for item in result['proposals'])
    assert not result['support_updated'] and not result['independent_confirmation']


def test_transfer_protects_comparison_from_checks_and_cache_does_not_hold_twice(tmp_path):
    agent, lab = agent_scene(tmp_path)
    agent.config = replace(agent.config, compute_budget=10000)
    candidate = {**edit_candidate(), 'version': 1}
    agent.theory_consolidation = {'candidates': {candidate['id']: candidate}}
    _save(agent)
    parent = binary('mul', constant(5.), feature())
    register_theories(agent, [theory('parent', parent)])
    checks = [{**item, 'context_ref': deepcopy(CONTEXT)} for item in examples(parent)]
    arguments = {'operator': 'theory_transfer', 'candidate_id': candidate['id'], 'theory_id': 'funded',
        'parent_theory_id': 'parent', 'context_ref': deepcopy(CONTEXT), 'output_unit': 'V',
        'allocation': {'compute': 5000, 'comparison_rows': 4}, 'evidence_checks': checks,
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}
    original_limit = agent.config.compute_budget
    before = agent.costs['compute_cost']
    result = agent._execute('run', deepcopy(arguments), lab)
    reserved = 4 * len(result['generated_theories'])
    assert result['reserved_comparison_compute'] == reserved > 0
    assert agent.config.compute_budget == original_limit - reserved
    assert result['new_compute_cost'] == agent.costs['compute_cost'] - before
    assert result['new_compute_cost'] + reserved <= 5000
    assert all(check['status'] in {'passed', 'failed'}
               for item in result['generated_theories'] for check in item['checks'])
    # The caller restores its outer configuration after routing. Previously
    # exported paths are now already included in its fixed complete-pool hold.
    agent.config = replace(agent.config, compute_budget=original_limit)
    paid = deepcopy(agent.costs)
    cached = agent._execute('run', arguments, lab)
    assert cached['cache_hit'] and cached['reserved_comparison_compute'] == 0
    assert agent.config.compute_budget == original_limit and agent.costs == paid
    assert_no_new_science(agent, lab)


def test_recall_holds_every_candidate_output_in_its_same_finite_slice(tmp_path):
    agent, lab = agent_scene(tmp_path)
    candidate = {**edit_candidate(), 'version': 1}
    other = {**deepcopy(candidate), 'id': 'another_saved_relation'}
    agent.theory_consolidation = {'candidates': {item['id']: item for item in (candidate, other)}}
    _save(agent)
    parent = binary('mul', constant(5.), feature())
    register_theories(agent, [theory('parent', parent)])
    question = agent._open_question({'id': 'finite_revision', 'question': 'Revise the measured fixture Program.',
        'context_ref': deepcopy(CONTEXT), 'theory_target': {
            'context_ref': deepcopy(CONTEXT), 'role_signature': deepcopy(ROLES), 'output_unit': 'V',
            'parent_theory_id': 'parent',
            'evidence_checks': [{**item, 'context_ref': deepcopy(CONTEXT)} for item in examples(parent)],
            'revision_fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}})
    arguments = {'allocation': {'compute': 400, 'comparison_rows': 4}}
    original_limit = agent.config.compute_budget
    result = recall_frozen_theories(agent, question, arguments)
    actual_count = sum(len(item['theory_ids']) for item in result['work'])
    assert actual_count > 0
    assert result['reserved_comparison_compute'] == 4 * actual_count
    assert result['new_compute_cost'] + result['reserved_comparison_compute'] <= 400
    assert agent.config.compute_budget == original_limit - result['reserved_comparison_compute']
    before = deepcopy(agent.costs)
    cached = recall_frozen_theories(agent, question, arguments)
    assert cached['cache_hit'] and cached['reserved_comparison_compute'] == 0
    assert agent.costs == before
    assert_no_new_science(agent, lab)
