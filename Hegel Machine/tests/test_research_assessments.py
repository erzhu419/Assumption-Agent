"""Common reuses complete paid native assessments without fitting or rescoring."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from hegel_machine.cognition.research_assessments import resolve_research_comparison
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.theory_consolidation import LIBRARY_PATH, _save
from test_frozen_theory_knowledge import recipient, refs
from test_question_routing import CONTEXT, future_question, probe, route, theory
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, constant, edit_candidate, feature
from test_theory_consolidation import agent_scene, assert_no_new_science


def fitted_scene(path):
    agent, lab = agent_scene(path)
    frozen = {**edit_candidate(), 'version': 1}
    agent.theory_consolidation = {'candidates': {frozen['id']: frozen}}
    _save(agent)
    parent = binary('mul', constant(5.), feature())
    square = binary('mul', feature(), feature())
    register_theories(agent, [theory('parent', parent), theory('old_square', square)])
    # These coarse point checks permit a research alternative with nonzero MSE.
    # They do not change Common's much stricter task/confirmation MSE criterion.
    checks = [{**item, 'tolerance': 6., 'context_ref': deepcopy(CONTEXT),
        'source_ref': {'file': str(agent.workspace / 'paid_fixture.json'), 'json_path': [index]}}
        for index, item in enumerate(examples(parent))]
    (agent.workspace / 'paid_fixture.json').write_text(json.dumps([item['observation'] for item in checks]))
    feedback = agent._execute('run', {'operator': 'theory_transfer', 'candidate_id': frozen['id'],
        'theory_id': 'fitted', 'parent_theory_id': 'parent', 'component_theory_ids': ['old_square'],
        'context_ref': deepcopy(CONTEXT), 'output_unit': 'V', 'allocation': {'compute': 5000},
        'evidence_checks': checks,
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}, lab)
    assert feedback['generated_theories']
    contract = {'maximum': 1, 'assessment_basis_ref': {'file': str(agent.workspace / LIBRARY_PATH),
        'json_path': ['executions', 'relation_transfer:fitted', 'request_basis']}}
    return agent, lab, contract


def test_exact_paid_assessments_are_reused_without_prediction_or_unobserved_scores(tmp_path, monkeypatch):
    agent, lab, contract = fitted_scene(tmp_path)
    before = deepcopy((agent.costs, agent.theory_index, agent.theory_consolidation))

    def forbidden(*args, **kwargs):
        raise AssertionError('Resolving an already paid score must not predict or fit')

    monkeypatch.setattr(Program, 'predict', forbidden)
    question = {**future_question(agent), 'research_comparison': contract}
    result = resolve_research_comparison(agent, question, resolved_theories(agent))
    assert result['status'] == 'ready' and result['row_count'] == 4
    assert result['assessed']
    assert {'parent', 'old_square'} <= set(result['mandatory_theory_ids'])
    for assessed in result['assessed']:
        assert assessed.reference_loss is None and assessed.fresh_loss is None
        assert assessed.evidence_loss == agent.theory_index[assessed.candidate.candidate_id][
            'consolidation_ref']['fitting']['training_loss']
        assert '"observation"' not in json.dumps(assessed.assessment_scope)
        assert assessed.assessment_scope['independent_confirmation'] is False
    assert (agent.costs, agent.theory_index, agent.theory_consolidation) == before
    assert_no_new_science(agent, lab)


@pytest.mark.parametrize('change', ['quantile', 'partial_rows', 'missing_observation'])
def test_incompatible_or_incomplete_score_scope_is_unknown_and_keeps_original_entries(tmp_path, change):
    agent, lab, contract = fitted_scene(tmp_path)
    original = agent.theory_consolidation['executions']['relation_transfer:fitted']['request_basis']
    changed = deepcopy(original)
    if change == 'quantile':
        changed['fitting']['risk_quantile'] = .8
    elif change == 'partial_rows':
        changed['evidence_checks'] = changed['evidence_checks'][:-1]
    else:
        changed['evidence_checks'][0].pop('observation')
    agent.theory_consolidation['executions']['different_scope'] = {'request_basis': changed}
    _save(agent)
    contract['assessment_basis_ref']['json_path'][1] = 'different_scope'
    theories = resolved_theories(agent)
    before = deepcopy(agent.costs)
    result = resolve_research_comparison(agent, {'research_comparison': contract}, theories)
    assert result['status'] == 'UNKNOWN' and result['assessed'] == []
    assert set(result['mandatory_theory_ids']) == {item['id'] for item in theories}
    assert result['unknown'] and agent.costs == before
    assert_no_new_science(agent, lab)


def test_changed_theory_version_cannot_inherit_an_earlier_program_score(tmp_path):
    agent, lab, contract = fitted_scene(tmp_path)
    identifier = next(item['id'] for item in resolved_theories(agent) if item.get('consolidation_ref'))
    declaration = deepcopy(agent.theory_index[identifier])
    declaration['program'] = constant(123.).to_dict()
    register_theories(agent, [declaration])
    result = resolve_research_comparison(agent, {'research_comparison': contract}, resolved_theories(agent))
    assert identifier not in {row.candidate.candidate_id for row in result['assessed']}
    assert identifier in result['mandatory_theory_ids']
    assert any(item['id'] == identifier for item in result['unknown'])
    assert_no_new_science(agent, lab)


def test_inherited_assessment_refs_keep_the_physical_source_and_do_not_add_support(tmp_path):
    source, source_lab, contract = fitted_scene(tmp_path / 'source')
    middle, middle_lab = recipient(tmp_path / 'middle', refs(source))
    _save(middle)
    target, target_lab = recipient(tmp_path / 'target', refs(middle))
    before = deepcopy(target.costs)
    result = resolve_research_comparison(target, {'research_comparison': contract}, resolved_theories(target))
    assert result['status'] == 'ready' and result['assessed']
    references = list(result['assessment_refs'].values())
    assert all(item['file'] == str(source.workspace / LIBRARY_PATH) for item in references)
    assert all(item['value']['fitted_rows'] == 4 for item in load_source_values(references, target.workspace))
    assert target.costs == before and not result['support_updated']
    assert all(not row.assessment_scope['independent_confirmation'] for row in result['assessed'])
    assert_no_new_science(target, target_lab)


def test_Common_route_uses_the_research_pool_in_its_actual_groups_and_reuses_it_after_restore(tmp_path):
    agent, lab, contract = fitted_scene(tmp_path)
    question = future_question(agent)
    experiments = [probe(agent, 0., 'zero'), probe(agent, 2., 'two')]
    plain = route(agent, lab, question=question, experiments=experiments)
    before_groups = len(plain['selection']['groups'])
    result = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}, 'research_comparison': contract}, lab)
    assert len(result['selection']['groups']) < before_groups
    assert result['selection']['scores'] != plain['selection']['scores']
    assert result['selection']['research_comparison']
    saved = next(item for item in agent.questions if item['id'] == question['id'])
    assert saved['research_comparison'] == contract
    assert 'research_comparison' in saved['routing_execution']['basis']
    costs = deepcopy(agent.costs)
    again = route(agent, lab, question={'id': question['id']})
    assert again['cache_hit'] and again['new_compute_cost'] == 0 and agent.costs == costs
    previous = agent.result('researching')
    cold, cold_lab = agent_scene(tmp_path / 'restored')
    cold_lab.costs = {'observation_cost': cold_lab.paid}
    restored = cold.restore_practice_research(previous, agent.workspace, cold_lab, confirmation_seed=0)
    assert restored['new_observations'] == 0
    cold_question = next(item for item in cold.questions if item['id'] == question['id'])
    assert cold_question['research_comparison'] == contract
    cold_costs = deepcopy(cold.costs)
    repeated = route(cold, cold_lab, question={'id': question['id']})
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0 and cold.costs == cold_costs
    assert agent.model is None and not agent.confirmations
    assert_no_new_science(agent, lab)
    assert_no_new_science(cold, cold_lab)


def test_explicit_route_roster_does_not_silently_add_other_fitted_programs(tmp_path):
    agent, lab, contract = fitted_scene(tmp_path)
    question = future_question(agent, 'exact_roster')
    question['candidate_theory_ids'] = ['parent']
    question['research_comparison'] = contract
    result = route(agent, lab, question=question, experiments=[probe(agent)])
    identifiers = {identifier for group in result['selection']['groups'] for identifier in group['theory_ids']}
    assert identifiers == {'parent'}
    assert result['selection']['research_comparison']['mandatory_theory_ids'] == ['parent']
    assert_no_new_science(agent, lab)


def overwritten_scene(path, *, mismatch=None, latest_matches=False):
    source, source_lab, contract = fitted_scene(path / 'source')
    original = source.theory_consolidation['executions']['relation_transfer:fitted']
    identifier = original['result']['generated_theories'][0]['theory_id']
    if mismatch:
        source.theory_consolidation['executions']['scope_basis'] = {
            'request_basis': deepcopy(original['request_basis'])}
        contract['assessment_basis_ref']['json_path'][1] = 'scope_basis'
        if mismatch == 'quantile':
            original['request_basis']['fitting']['risk_quantile'] = .8
        elif mismatch == 'version':
            original['result']['generated_theories'][0]['version'] += 1
        else:
            original['request_basis']['evidence_checks'].reverse()
        _save(source)
    middle, middle_lab = recipient(path / 'delta', refs(source))
    updated = middle.theory_consolidation['executions']['relation_transfer:fitted']
    updated['allowance'] += 1
    if not latest_matches:
        updated['result']['generated_theories'] = []
    _save(middle)
    target, target_lab = recipient(path / 'target', refs(middle))
    return source, middle, target, target_lab, contract, identifier


def test_overwritten_execution_uses_the_exact_old_transfer_source_without_prediction(tmp_path, monkeypatch):
    source, middle, target, lab, contract, identifier = overwritten_scene(tmp_path)
    costs = deepcopy(target.costs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Recovering retained assessments must not predict')

    monkeypatch.setattr(Program, 'predict', forbidden)
    result = resolve_research_comparison(target, {'research_comparison': contract}, resolved_theories(target))
    restored = next(row for row in result['assessed'] if row.candidate.candidate_id == identifier)
    assert restored.assessment_scope['execution_basis_ref'] == {
        'file': str(source.workspace / LIBRARY_PATH),
        'json_path': ['executions', 'relation_transfer:fitted', 'request_basis']}
    assert restored.assessment_scope['assessment_ref']['file'] == str(source.workspace / LIBRARY_PATH)
    assert restored.reference_loss is None and restored.fresh_loss is None
    assert identifier not in result['mandatory_theory_ids']
    assert target.costs == costs
    assert_no_new_science(target, lab)


@pytest.mark.parametrize('mismatch', ['quantile', 'version', 'evidence_order'])
def test_old_physical_execution_requires_its_exact_quantile_version_and_ordered_rows(tmp_path, mismatch):
    source, middle, target, lab, contract, identifier = overwritten_scene(tmp_path, mismatch=mismatch)
    before = deepcopy(target.costs)
    result = resolve_research_comparison(target, {'research_comparison': contract}, resolved_theories(target))
    assert identifier not in {row.candidate.candidate_id for row in result['assessed']}
    assert identifier in result['mandatory_theory_ids']
    assert any(row['id'] == identifier and row['reason'] == 'matching_paid_fit_execution_required'
               for row in result['unknown'])
    assert target.costs == before
    assert_no_new_science(target, lab)


def test_matching_current_execution_remains_preferred_to_older_source(tmp_path):
    source, middle, target, lab, contract, identifier = overwritten_scene(tmp_path, latest_matches=True)
    result = resolve_research_comparison(target, {'research_comparison': contract}, resolved_theories(target))
    row = next(row for row in result['assessed'] if row.candidate.candidate_id == identifier)
    assert row.assessment_scope['execution_basis_ref']['file'] == str(middle.workspace / LIBRARY_PATH)
    assert_no_new_science(target, lab)
