"""Measured native research befores do not replace opaque adopted predictors."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import revision_relations
from hegel_machine.cognition.language import Program
from scripts.run_anonymous_revision_inquiry import ORGANIZATION_ORIGIN
from test_inquiry_revision import action
from test_registered_revision_target import revision_scene, forbid_acquisition


def opaque_scene(tmp_path):
    agent, lab, owner = revision_scene(tmp_path)
    agent._execute('evaluate', {'model': 'archive/failed.py'}, lab)
    return agent, lab, owner


def test_measured_research_before_reaches_real_proposer_without_adopting_or_recomputing_parent(
        tmp_path, monkeypatch):
    agent, lab, owner = opaque_scene(tmp_path)
    round_ = owner['native_condition_research'][0]
    registered = round_['native_theory_registration']['registered']
    first = registered[0]
    bank = round_['candidates']
    # The policy follows registration order, not an artificially favorable loss
    # or the learned object's pattern match. Both are still measured failures.
    assert bank[0]['fit_mse'] > agent.config.confirmation_mse_threshold
    assert bank[0]['fit_mse'] > bank[1]['fit_mse']
    before = deepcopy((agent.evaluated_model, agent.costs, bank, agent.theory_index))
    forbid_acquisition(lab, monkeypatch)
    with monkeypatch.context() as binding_only:
        binding_only.setattr(Program, 'predict', lambda *args: pytest.fail('Binding must not predict'))
        agent._bind_revision_target(owner)
    assert (agent.evaluated_model, agent.costs, bank, agent.theory_index) == before
    target = owner['theory_target']
    assert target['parent_theory_id'] == first['theory_id']
    assert target['parent_binding_role'] == 'research_candidate_before'
    evidence = target['parent_selection_evidence']
    assert evidence['selection_rule'] == 'registered_round_order_first_complete_paid_failure'
    assert evidence['rows'] == len(agent.rows) == 1
    assert evidence['records'] == len(agent.observations) == 1
    assert evidence['fit_mse_ref']['json_path'][-3:] == ['candidates', 0, 'fit_mse']
    assert evidence['fit_evidence_ref']['json_path'][-1] == 'fit_evidence'
    assert evidence['incumbent_source_model'] == 'archive/failed.py'
    assert all('observation' not in check for check in target['evidence_checks'])
    calls = []
    propose = revision_relations.propose_revision

    def observed(candidate, parent, components, **kwargs):
        result = propose(candidate, parent, components, **kwargs)
        calls.append((deepcopy(parent), deepcopy(kwargs['evidence_checks']), result))
        return result

    monkeypatch.setattr(revision_relations, 'propose_revision', observed)
    agent._active_question_id = owner['id']
    agent._before_model_turn(lab)
    assert len(calls) == 1
    parent, checks, proposed = calls[0]
    assert parent['id'] == first['theory_id'] and parent['program'] == bank[0]['program']
    assert len(checks) == 1 and checks[0]['observation']['target'] == 1.
    assert proposed['parent_binding']['component_id'] == first['theory_id']
    assert proposed['proposals']
    assert all(proposal['fitting']['fitted_rows'] == 1 for proposal in proposed['proposals'])
    assert agent.evaluated_model == before[0]
    assert agent.costs['observation_cost'] == before[1]['observation_cost']
    assert agent.costs['model_calls'] == 0 and not agent.confirmations
    assert agent.knowledge['finite_native_author']['predictor_paths'] == ['archive/failed.py']
    event = next(item for item in reversed(agent.events) if item.get('origin') == ORGANIZATION_ORIGIN)
    assert event['feedback']['frozen_recall']['work']
    repeated_costs = deepcopy(agent.costs)
    agent._active_question_id = owner['id']
    agent._before_model_turn(lab)
    assert len(calls) == 1 and agent.costs == repeated_costs
    agent._write_evidence()
    # This is the existing checkpoint path, not a new provenance sidecar.
    import json
    saved = json.loads((agent.workspace / 'research_state.json').read_text())
    recovered = next(q for q in saved['questions'] if q['id'] == owner['id'])['theory_target']
    assert recovered['parent_selection_evidence'] == evidence
    assert recovered['parent_binding_role'] == 'research_candidate_before'


@pytest.mark.parametrize('missing', [
    'no_candidates', 'old_paid_scope', 'partial_fit', 'partial_records',
    'different_meaning', 'different_context', 'different_unit', 'different_role',
    'changed_version', 'wrong_program_ref', 'no_measured_failure',
])
def test_incomplete_or_different_scope_does_not_become_a_measured_research_parent(
        tmp_path, monkeypatch, missing):
    agent, lab, owner = opaque_scene(tmp_path)
    round_ = owner['native_condition_research'][0]
    before_round = deepcopy(round_)
    if missing == 'no_candidates':
        round_['native_theory_registration']['registered'] = []
    elif missing == 'old_paid_scope':
        agent._execute('query', {'actions': [action(2.)]}, lab)
    elif missing == 'partial_fit':
        for candidate in round_['candidates']:
            candidate['fit_evidence'] = []
    elif missing == 'partial_records':
        round_['fit_record_indices'] = []
    elif missing == 'no_measured_failure':
        for candidate in round_['candidates']:
            candidate['fit_mse'] = 0.
    else:
        for item in round_['native_theory_registration']['registered']:
            theory = agent.theory_index[item['theory_id']]
            if missing == 'different_meaning':
                theory['source_ref']['meaning_version'] += 1
            elif missing == 'different_context':
                theory['context_ref'] = {'source_id': 'another_target'}
            elif missing == 'different_unit':
                theory['output_unit'] = 'other'
            elif missing == 'different_role':
                theory['role_signature'] = {'u': {'role': 'unrelated', 'unit': 'V', 'time': 'current'}}
            elif missing == 'changed_version':
                theory['version'] += 1
            else:
                theory['program_source_ref']['json_path'][-1] = 'not_this_program'
    before = deepcopy((agent.evaluated_model, agent.costs))
    forbid_acquisition(lab, monkeypatch)
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('Binding must not predict'))
    agent._bind_revision_target(owner)
    target = owner['theory_target']
    assert 'parent_theory_id' not in target and 'parent_binding_role' not in target
    assert 'parent_selection_evidence' not in target and 'evidence_checks' not in target
    assert target['parent_binding_obligation']['reason'] == 'unique_retained_native_parent_required'
    assert (agent.evaluated_model, agent.costs) == before
    if missing == 'old_paid_scope':
        assert round_ == before_round


def test_new_paid_rows_revoke_automatic_research_before_without_relabeling_old_failure(tmp_path):
    agent, lab, owner = opaque_scene(tmp_path)
    agent._bind_revision_target(owner)
    assert owner['theory_target']['parent_binding_role'] == 'research_candidate_before'
    round_ = deepcopy(owner['native_condition_research'][0])
    agent._execute('query', {'actions': [action(2.)]}, lab)
    agent._bind_revision_target(owner)
    assert 'parent_theory_id' not in owner['theory_target']
    assert 'parent_binding_role' not in owner['theory_target']
    assert 'evidence_checks' not in owner['theory_target']
    assert 'paid_evidence_scope' not in owner['theory_target']
    assert owner['native_condition_research'][0] == round_
    assert agent.evaluated_model['source_model'] == 'archive/failed.py'
