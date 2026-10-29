"""Prepared repairs can reuse exact scores without restoring target decisions."""

from copy import deepcopy
import json

import pytest

from scripts import prepare_pr3_source_successor as preparation
from test_inquiry_revision import action
from test_python_paid_score_reuse import ChoiceLab, MODELS, choose, no_execution, scene


INDEX = '_paid_python_scores/_evaluated_model/assessments.json'
QUESTION = {'id': 'q', 'question': 'Predict the public response',
            'context_ref': {'apparatus': 'local_fixture'}}


def prepared_scene(tmp_path, *, include_reference=True):
    source, lab = scene(tmp_path / 'paid_source')
    choose(source, lab)
    reference = {'file': str(tmp_path / 'initial_paid_scores.json'), 'json_path': []}
    # This is a pre-frozen numerical reference index, not the old target result.
    index = json.loads((source.workspace / INDEX).read_text())
    index['contexts'] = []
    preparation._write(reference['file'], index)
    plan = {'prepared_public_questions': {name: deepcopy(QUESTION)
                                         for name in ('case_000', 'case_001')}}
    if include_reference:
        plan['paid_score_index_refs_by_case'] = {'case_000': reference}
    library = tmp_path / 'empty_library.json'
    preparation._write(library, {'candidates': {}})
    organization = {'phase': 'target', 'theories': [], 'initial_library_path': str(library),
                    'pr3_library_path': str(library),
                    'exploration_policy': {'kind': 'seeded_legal', 'seed': 11}}
    knowledge = {'files': deepcopy(source.files), 'finite_native_author': {}}
    factory, _ = preparation._prepared_callbacks(plan)

    def fresh(case='case_000', arm='fixed_pr2'):
        agent = factory(None, tmp_path / case / arm, arm=arm,
            research_organization=organization, knowledge=deepcopy(knowledge),
            source_task=False, config=source.config)
        assert not agent.rows and not agent.observations and not agent.events
        assert agent.evaluated_model is None and agent.model is None
        assert not agent.revisions and not agent.confirmations
        local_lab = ChoiceLab()
        agent._execute('query', {'actions': [action(1., 2.)],
            'question': deepcopy(plan['prepared_public_questions'][case])}, local_lab)
        return agent, local_lab

    return source, plan, index, fresh


def test_registered_arms_share_exact_paid_refs_and_reapply_current_retention(tmp_path, monkeypatch):
    source, plan, index, fresh = prepared_scene(tmp_path)
    originals = [(source.workspace / INDEX).read_bytes(), deepcopy(source.evaluated_model),
                 deepcopy(source.costs)]
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    for arm in ('archive', 'fixed_pr2', 'pr3'):
        agent, lab = fresh(arm=arm)
        assert json.loads((agent.workspace / INDEX).read_text()) == {
            **index, 'historical_score_index_ref': plan['paid_score_index_refs_by_case']['case_000']}
        first = choose(agent, lab)
        assert first['reused_score_count'] == 2 and first['new_scored_count'] == 0
        assert first['evaluations'][1]['retention_decision']['status'] == 'candidate_retained'
        assert agent.costs['compute_cost'] == 0
        for evaluation in first['evaluations']:
            reuse = evaluation['paid_score_reuse']
            assert reuse['original_costs']['compute_cost'] == len(source.rows)
            assert reuse['additional_scoring_compute'] == 0
            assert reuse['execution_basis_ref'] in index['assessments']
        second = choose(agent, lab)
        assert second['evaluations'][1]['retention_decision']['status'] == 'incumbent_retained'
        assert agent.evaluated_model['source_model'] == MODELS[1]
    assert (source.workspace / INDEX).read_bytes() == originals[0]
    assert source.evaluated_model == originals[1] and source.costs == originals[2]


@pytest.mark.parametrize('change', ['row_order', 'source'])
def test_reference_mismatch_uses_the_ordinary_paid_score(tmp_path, change):
    _, _, _, fresh = prepared_scene(tmp_path)
    agent, lab = fresh()
    if change == 'row_order':
        agent.rows.reverse()
    else:
        (agent.workspace / MODELS[0]).write_text('def predict(row): return -1.\n')
    result = choose(agent, lab, MODELS[:1])
    assert result['new_scored_count'] == 1 and result['reused_score_count'] == 0
    assert agent.costs['compute_cost'] == len(agent.rows)


@pytest.mark.parametrize('include_reference,case', [(True, 'case_001'), (False, 'case_000')])
def test_case_isolation_and_absent_reference_leave_ordinary_path(tmp_path, include_reference, case):
    _, _, _, fresh = prepared_scene(tmp_path, include_reference=include_reference)
    agent, lab = fresh(case=case)
    assert not (agent.workspace / INDEX).exists()
    result = choose(agent, lab)
    assert result['new_scored_count'] == 2 and result['reused_score_count'] == 0
    assert agent.costs['compute_cost'] == 2 * len(agent.rows)
