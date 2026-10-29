"""Ordinary return induction reuses supported methods or emits a real pair input."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.method_action_research import (
    read_method_reference, review_python_method_returns, resume_python_method_proposals,
)
from hegel_machine.cognition.method_feedback import finalize_python_method_feedback
from hegel_machine.cognition.research_programs import export_research_selection_program
from hegel_machine.cognition.theory_consolidation import install_python_research_method
from scripts import run_state_method_pairing as study
from test_automatic_method_content_proposal import append_source, review, save, setup_scene
from test_state_method_pairing import paid_fixture


def proposed_scene(path):
    agent, lab, owner, p0, previous = setup_scene(path)
    append_source(agent, p0, 'success')
    second = append_source(agent, p0, 'missed', missed=True)
    question_id = second['method_question_refs'][0]['question_id']
    result = review(agent, lab, question_id)['method_action_research']
    question = next(item for item in agent.questions if item['id'] == question_id)
    return agent, lab, owner, previous, question, result


def supported_equivalent(agent, owner, previous, result, *, changed_source=False,
                         complete=True, ordinary_path=None):
    plan = deepcopy(result['proposals']['candidates'][0]['research_selection_plan'])
    plan['source_refs'] = [{'file': 'different_provenance_only.json'}]
    if ordinary_path is not None:
        plan['ordinary_method_path'] = ordinary_path
    if changed_source:
        agent._save('different_ordinary.py',
            (agent.workspace / plan['ordinary_method_path']).read_text() + '\n# changed dependency\n')
        plan['ordinary_method_path'] = 'different_ordinary.py'
    candidate = install_python_research_method(agent, candidate_id='already_measured',
        source=export_research_selection_program(plan), selection_plan=plan,
        source_refs=[], selection_ref=previous['selection_ref'],
        method_question_ref={'question_id': owner['id'], 'meaning_version': 1})
    owner['candidate_method_refs'].append({'id': candidate['id'], 'version': 1})
    outcome = save(agent.workspace, 'measured_outcome.json', {
        'method_ref': {'id': candidate['id'], 'version': 1},
        'status': 'completed', 'execution_complete': True, 'completed': complete,
        'quality': {'loss': 0 if complete else .1, 'threshold': 1e-10},
        'comparison': {'models_requested': 48, 'models_completed': 48, 'truncated': False},
        'costs': {'compute_cost': 400}})
    binding = {'candidate_id': candidate['id'], 'candidate_version': 1,
        'origin_ref': save(agent.workspace, 'measured_invocation.json', {'method': candidate['id']})}
    finalize_python_method_feedback(agent, candidate, binding,
        {'profile': {'residual': {'state': 'unresolved'}},
         'question_ref': {'question_id': 'public', 'meaning_version': 1}}, result_ref=outcome)
    return candidate, outcome


def test_supported_same_topic_body_reuses_frozen_dependency_and_real_return(tmp_path):
    agent, lab, owner, previous, question, first = proposed_scene(tmp_path)
    candidate, support = supported_equivalent(agent, owner, previous, first)
    before = deepcopy(agent.theory_consolidation['candidates'])
    result = review_python_method_returns(agent, question, public_spec=lab.public_spec(), compute_limit=1000)
    assert result['status'] == 'complete_supported_method_retained'
    assert result['installed_candidate'] == candidate
    assert result['existing_method_support_refs'] == [support]
    assert result['existing_support_scope'] == 'actual_completed_invocation'
    assert result['next_action']['arguments']['research_question']['id'] == 'public'
    assert result['next_action']['arguments']['path'] == candidate['execution']['path']
    assert not result['independent_confirmation'] and not result['support_updated']
    assert agent.theory_consolidation['candidates'] == before and lab.paid == 0


@pytest.mark.parametrize('defect', ['different_source', 'failed_return', 'different_topic'])
def test_plan_text_alone_or_guard_support_does_not_validate_method_body(tmp_path, defect):
    agent, lab, owner, previous, question, first = proposed_scene(tmp_path)
    candidate, _ = supported_equivalent(agent, owner, previous, first,
        changed_source=defect == 'different_source', complete=defect != 'failed_return')
    if defect == 'different_topic':
        owner['candidate_method_refs'] = [ref for ref in owner['candidate_method_refs']
                                          if ref['id'] != candidate['id']]
    result = review_python_method_returns(agent, question, public_spec=lab.public_spec(), compute_limit=1000)
    assert result['status'] == 'method_content_proposed_pending_validation'
    assert result['installed_candidate'] is None and result['next_action'] is None
    assert result['validation_requests'][0]['method_pair_ref'] == first['validation_requests'][0]['method_pair_ref']
    assert len(list((agent.workspace / 'method_feedback').glob('content_proposal_*.json'))) == 1
    assert lab.paid == 0


def test_new_body_bundle_is_consumed_by_existing_paid_pair_quote_without_execution(tmp_path, monkeypatch):
    agent, lab, _, previous, _, result = proposed_scene(tmp_path / 'method')
    request, = result['validation_requests']
    bundle = read_method_reference(request['method_pair_ref'], agent.workspace)
    assert bundle['status'] == 'unmeasured_local_research_intervention'
    assert not bundle['source_support_inherited'] and not request['execution_authorized']
    assert bundle['cells']['00']['selection_plan']['when_false'] == previous['selection_plan']['when_false']
    assert bundle['cells']['10']['selection_plan']['when_true'] == previous['selection_plan']['when_true']
    paid, case, values = paid_fixture(tmp_path / 'paid')

    def no_science(*args, **kwargs):
        raise AssertionError('method proposal preparation cannot fit or open evaluation labels')

    monkeypatch.setattr(study, 'AnonymousHistoryLab', no_science)
    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_state_claims', no_science)
    prepared = study.prepare(paid, tmp_path / 'prepared', case_ref=case, evaluation_seed=402,
        research_compute_cap=10000, compute_cap=30000, evaluation_rows=20,
        method_pair_ref=request['method_pair_ref'])
    assert prepared['quote']['status'] == 'quoted'
    assert not prepared['execution_authorized']
    assert prepared['cells']['10']['method_ref']['id'] == request['candidate_id']
    assert prepared['cells']['10']['selection_plan']['when_false']['first_segment'][
        'parent_binding'] == 'paid_startup_witnesses'
    assert prepared['finite_native_author']['max_candidates'] == 4
    assert prepared['paid_scene']['expected_rows'] == sum(len(record['rows']) for record in values['records'])
    factories = study.make_factories(prepared, tmp_path / 'paired')
    for name, cell in prepared['cells'].items():
        restored = factories['restore_cell'](name, cell, tmp_path / 'paired' / name)
        assert (restored['agent'].workspace / restored['path']).is_file()
        assert restored['agent'].costs['observation_cost'] == 0
    assert bundle['cells']['00']['id'] != previous['id']
    assert bundle['cells']['00']['previous_method_ref'] == {'id': previous['id'], 'version': 1}
    assert len(agent.theory_consolidation['candidates']) == 2 and lab.paid == 0


def test_resolution_exhaustion_keeps_unmeasured_proposal_and_does_not_adopt(tmp_path):
    agent, lab, _, _, question, _ = proposed_scene(tmp_path)
    result = review_python_method_returns(agent, question, public_spec=lab.public_spec(),
        compute_limit=0)
    assert result['status'] == 'scoped_unknown'
    assert result['reason'] == 'method_proposal_resolution_not_funded'
    assert result['installed_candidate'] is None and result['next_action'] is None
    assert result['proposals']['candidates'][0]['status'] == 'unmeasured_local_research_intervention'
    assert lab.paid == 0


def test_pending_induction_resumes_against_registered_current_adapter_without_reinduction(tmp_path, monkeypatch):
    agent, lab, owner, p0, previous = setup_scene(tmp_path)
    current = (agent.workspace / 'ordinary.py').read_text()
    old = current.replace("        if requested.get('parent_binding') == 'paid_startup_witnesses':\n"
                          "            action = _bind_paid_startup_witnesses(action, memory)\n", '')
    assert old != current
    agent._save('historical_ordinary.py', old)
    plan = {**previous['selection_plan'], 'ordinary_method_path': 'historical_ordinary.py'}
    historical = install_python_research_method(agent, candidate_id='historical',
        source=export_research_selection_program(plan), selection_plan=plan,
        source_refs=[], selection_ref=previous['selection_ref'])
    owner['candidate_method_refs'] = [{'id': historical['id'], 'version': 1}]
    owner['method_use_plan']['candidate_id'] = historical['id']
    append_source(agent, p0, 'success')
    returned = append_source(agent, p0, 'missed', missed=True)
    question = next(item for item in agent.questions
                    if item['id'] == returned['method_question_refs'][0]['question_id'])
    pending = review(agent, lab, question['id'])['method_action_research']
    assert pending['status'] == 'method_content_proposed_pending_validation'
    candidate, support = supported_equivalent(agent, owner, historical, pending,
                                              ordinary_path='ordinary.py')
    preserved = deepcopy((pending, agent.theory_consolidation['candidates']))
    agent.knowledge['finite_native_author'] = {'question': {'id': 'public', 'meaning_version': 1}}

    def no_induction(*args, **kwargs):
        raise AssertionError('saved paid induction must not be repeated')

    monkeypatch.setattr('hegel_machine.cognition.research_programs.derive_paid_witness_binding_revision', no_induction)
    result = review(agent, lab, question['id'])['method_action_research']
    assert result['status'] == 'complete_supported_method_retained'
    assert result['installed_candidate'] == candidate
    assert result['existing_method_support_refs'] == [support]
    assert result['proposal_analysis_reused'] and result['new_compute_cost'] <= 10
    dependency = result['proposals']['candidates'][0]['ordinary_dependency_revision']
    assert dependency['previous_source_ref']['file'].endswith('/shared/historical_ordinary.py')
    assert dependency['current_source_ref']['file'] == str(agent.workspace / 'ordinary.py')
    assert (pending, agent.theory_consolidation['candidates']) == preserved and lab.paid == 0


def test_current_dependency_binding_requires_actual_finite_entry_not_just_filename(tmp_path):
    agent, lab, _, _, question, pending = proposed_scene(tmp_path)
    agent.knowledge['finite_native_author'] = {'question': {'id': 'public'}}
    agent._save('ordinary.py', 'def select_actions(*args): return {}\n')
    result = resume_python_method_proposals(agent, question, pending, compute_limit=10)
    assert result['status'] == 'scoped_unknown'
    assert result['reason'] == 'current_registered_finite_ordinary_source_required'
    assert result['next_action'] is None and result['installed_candidate'] is None
    assert lab.paid == 0
