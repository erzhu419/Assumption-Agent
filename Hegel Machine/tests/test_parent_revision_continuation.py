"""Answered measurements and parent revision are separate, ordinary research work.

Numerical rows/formulas below are existing local fixtures, not D3 scientific data.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.finite_native_author import export_finite_native_author
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.theory_consolidation import LIBRARY_PATH
from test_finite_native_author import POLICY, declaration
from test_question_routing import CONTEXT, ROLES, scene, theory, route
from test_revision_context_fitting import FITTING
from test_revision_relations import binary, constant, edit_candidate, feature, revision


def revision_scene(tmp_path):
    agent, lab = scene(tmp_path)
    public = lab.public_spec()
    lab.public_spec = lambda: {**public, 'max_probe_length': 24, 'feature_bounds': {'u': [-10., 10.]}}
    before = [binary('mul', constant(value), feature()) for value in (2., 3.)]
    learned = edit_candidate([revision(str(i), binary('add', parent, constant(i + 1)),
        before_program=parent.to_dict(), counterexamples=[{
            'source_ref': {'file': 'paid_source_fixture.json', 'json_path': [i]},
            'observation': {'features': {'u': 2.}, 'target': 2 * (i + 2) + i + 1, 'history': []},
            'tolerance': 0.}]) for i, parent in enumerate(before)])
    candidates = {f'learned_{i}': {**deepcopy(learned), 'id': f'learned_{i}', 'version': 1}
                  for i in range(5)}
    agent.theory_consolidation = {'candidates': candidates, 'executions': {}}
    agent._save(LIBRARY_PATH, json.dumps(agent.theory_consolidation))
    agent._save('old.py', 'def predict(row): return 0.\n')
    agent._save('current.py', "def predict(row): return .5 * row['features']['u']\n")
    agent._save(POLICY, export_finite_native_author())
    register_theories(agent, [theory('old_parent', constant(0)),
                             theory('current_parent', binary('mul', constant(.5), feature()))])
    record = agent.observations[0]
    row = record['rows'][1]
    question = declaration()['question'] | {'kind': 'value', 'answer_type': 'number',
        'answer_resolution_ref': {'atol': 0.}, 'query_semantics': {'mode': 'model_query',
            'query_expression': {'op': 'Readout', 'record_index': 0, 'step': 1,
                'target': 'response', 'units': 'V', 'context_ref': deepcopy(CONTEXT),
                'binding': {'action': deepcopy(record['action']), 'features': deepcopy(row['features']),
                            'history': deepcopy(list(row['history']))}}},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
    question['theory_target'].update(parent_theory_id='old_parent',
        revision_fitting={**FITTING, 'operations': ['refit'], 'max_structures': 1},
        evidence_checks=[{'source_ref': {'file': 'laboratory_observations.json', 'json_path': [0, 'rows', i]},
            'context_ref': deepcopy(CONTEXT), 'observation': deepcopy(value), 'tolerance': 0.}
            for i, value in enumerate(record['rows'])])
    agent._execute('evaluate', {'model': 'old.py'}, lab)
    answer = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
    assert answer.get('answer', {}).get('value') == 1., answer
    assert answer['parent_revision_completed'] is False
    child = next(q for q in agent.questions if q['id'] == question['id'])
    agent._execute('evaluate', {'model': 'current.py'}, lab)
    assert agent.evaluated_model['source_model'] == 'current.py'
    child['theory_target']['parent_theory_id'] = 'current_parent'
    contract = declaration() | {'question': deepcopy(question), 'predictor_paths': ['old.py', 'current.py']}
    contract['question']['theory_target'] = deepcopy(child['theory_target'])
    agent.knowledge['finite_native_author'] = contract
    return agent, lab, child, candidates


def test_answered_readout_keeps_its_identity_while_five_objects_revise_the_current_parent(tmp_path):
    agent, lab, child, candidates = revision_scene(tmp_path)
    measured = deepcopy({key: child[key] for key in ('meaning_version', 'query_semantics', 'semantic_answer')})
    paid, actions = lab.paid, deepcopy(lab.actions)
    result = route(agent, lab, question={'id': child['id']})
    assert 'parent_revision_question_ref' in result
    obligation = next(q for q in agent.questions if q['id'] == result['question_id'])
    assert obligation['id'] != child['id'] and obligation['answer_type'] == 'program'
    assert obligation['query_semantics'] == {'mode': 'solution_relation'}
    assert obligation['answer_resolution_ref'] == {'relation': 'le',
        'atol': agent.config.confirmation_mse_threshold,
        'scope': 'paid_candidate_comparison_not_independent_confirmation'}
    assert obligation['theory_target']['parent_theory_id'] == 'current_parent'
    assert obligation['native_research_ref'] == {'question_id': child['id'], 'meaning_version': 1}
    work = result['frozen_recall']['work']
    assert {identifier for item in work for identifier in
            item.get('shared_candidate_ids', [item['candidate_id']])} == set(candidates)
    assert all(item.get('reason') != 'before_pattern_does_not_match_parent' for item in work)
    assert result['parent_revision_exports']['new_exports'] > 0
    assert agent.candidate_paths
    assert {key: child[key] for key in measured} == measured
    assert next(q for q in agent.questions if q['id'] == 'task')['status'] == 'reopened'
    assert lab.paid == paid and lab.actions == actions and not agent.confirmations
    assert agent.costs['model_calls'] == 0 and agent.model is None


def test_same_parent_and_evidence_reuse_completed_work_without_research_or_export(tmp_path, monkeypatch):
    from hegel_machine.cognition import theory_consolidation as consolidation
    agent, lab, child, _ = revision_scene(tmp_path)
    # A complete visit finishes one binding, not the entire finite frontier.
    # This fixture has three positions shared by its five identical objects.
    first = None
    for layer in range(3):
        current = route(agent, lab, question={'id': child['id']})
        first = first or current
        recall = current['frozen_recall']
        assert not recall['cache_hit']
        assert len(recall['complete_research_visits']) == 1
        visit = recall['complete_research_visits'][0]
        assert visit['status'] == 'complete_visit_executed'
        assert visit['search_position'] == {'layer': layer, 'offset': 1}
        assert visit['new_compute_cost'] <= visit['minimum_resources']['compute_cost']
        assert visit['allocated_resources']['compute_cost'] >= visit['quote']['compute']
        executions = [value for key, value in agent.theory_consolidation['executions'].items()
                      if key.startswith('relation_transfer:')]
        assert len(executions) == 1
        cursor = executions[0]['cursor']
        assert (cursor['layer'], cursor['offset'], cursor['pending_fits']) == (layer, 1, [])
        assert recall['search_completed'] is (layer == 2)
        assert executions[0]['search_completed'] is (layer == 2)
        assert agent.costs['compute_cost'] <= agent.config.compute_budget
    costs, exports = deepcopy(agent.costs), list(agent.candidate_paths)
    monkeypatch.setattr(consolidation, 'transfer_theory',
        lambda *a, **k: pytest.fail('Completed same-parent work must be reused'))
    repeated = route(agent, lab, question={'id': child['id']})
    assert repeated['cache_hit'] and repeated['frozen_recall']['cache_hit']
    assert repeated['new_compute_cost'] == repeated['parent_revision_exports']['new_exports'] == 0
    assert agent.costs == costs and agent.candidate_paths == exports
    assert repeated['parent_revision_question_ref'] == first['parent_revision_question_ref']


def test_native_author_preserves_original_bank_and_compares_the_complete_old_and_revision_pool(tmp_path):
    agent, lab, child, _ = revision_scene(tmp_path)
    bank = {'round_index': 0, 'status': 'structural_mismatch', 'observed_record_count': 1,
            'exported_models': [],
            'fit_record_indices': [0], 'fit_evidence': [0, 1], 'candidates': [
                {'model': 'old.py', 'program': constant(0).to_dict()},
                {'model': 'current.py', 'program': binary('mul', constant(.5), feature()).to_dict()}],
            'proposed_experiments': []}
    child['native_condition_research'] = [deepcopy(bank)]
    agent.operator_results.append({'operator': 'native_condition_research',
        'arguments': {'operator': 'native_condition_research', 'research_question': {'id': child['id']}},
        'research_question_id': child['id'], 'question_meaning_version': 1,
        'native_condition_research': deepcopy(bank)})
    from hegel_machine.cognition.temporal_paired_method import execute_procedure
    execute_procedure(agent, lab, POLICY, max_steps=1, allow_confirmation=False, record_events=True)
    old_result = deepcopy(agent.method_results[-1])
    assert old_result['feedback']['research_question_id'] == child['id']
    snapshot = agent.result('researching')['method_results']
    snapshot_before = deepcopy(snapshot)
    # The improving linear binding is the third complete visit. Comparing
    # after only the first constant binding correctly keeps the old parent.
    for _ in range(3):
        routed = route(agent, lab, question={'id': child['id']})
        assert routed['parent_revision_exports']['new_exports']
    assert routed['frozen_recall']['search_completed']
    expected = ['old.py', 'current.py', *agent.candidate_paths]
    from hegel_machine.cognition.temporal_paired_method import execute_procedure
    execute_procedure(agent, lab, POLICY, max_steps=1, allow_confirmation=False, record_events=True)
    choice = agent.method_results[-1]
    assert choice['proposal']['arguments']['operator'] == 'research_choice'
    assert choice['proposal']['arguments']['models'] == expected
    assert [item['path'] for item in choice['feedback']['evaluations']] == expected
    assert choice['feedback']['research_question_id'] == child['id']
    # Paid payloads now remain addressed by the central history module. Verify
    # their complete recovered values, including the old occurrence decision.
    from hegel_machine.cognition.paid_score_history import restore_score_history_value
    from hegel_machine.cognition.research_episodes import load_source_values
    restored = deepcopy(agent.method_results[-2])
    assert restored['feedback'].pop('paid_score_history') == 'assessment_references'
    restored['feedback']['evaluations'] = [restore_score_history_value(
        value, lambda reference: load_source_values([reference], agent.workspace)[0]['value'])
        for value in restored['feedback']['evaluations']]
    # JSON transport preserves histories as arrays rather than Python tuples.
    assert json.loads(json.dumps(restored)) == json.loads(json.dumps(old_result))
    assert snapshot == snapshot_before
    snapshot[-1]['feedback']['evaluations'][0]['retention_decision']['status'] = 'snapshot_only'
    assert agent.method_results[-2]['feedback']['evaluations'][0]['retention_decision']['status'] != 'snapshot_only'
    assert child['native_condition_research'] == [bank]
    assert child['meaning_version'] == 1 and child['semantic_answer']['value'] == 1.
    from hegel_machine.cognition.research_episodes import extract_episodes, retained_revision_sources
    episodes = extract_episodes(agent)['episodes']
    revised = next(item for item in episodes if item['episode_id'] == routed['question_id'] + '@meaning1')
    changes = revised['actual_changes']
    assert changes and all(item['kind'] == 'retained_program_replaced' for item in changes)
    assert all(item['fit_state_refs']['after']['consumer_question_ref']['json_path'] ==
               ['questions', agent.questions.index(child)] for item in changes)
    retained = retained_revision_sources(agent)
    assert retained
    owner = next(q for q in agent.questions if q['id'] == routed['question_id'])
    next_subject = deepcopy(owner['subject_refs'])
    next_subject[0]['version'] += 1
    agent._open_question({'id': owner['id'], 'subject_refs': next_subject})
    assert owner['meaning_version'] == 2
    # A new parent does not move the already executed revision to the new meaning.
    historical = retained_revision_sources(agent)
    assert set(historical) == set(retained)
    assert all(item['episode_ref']['meaning_version'] == 1 for item in historical.values())
    assert owner['meaning_history'][0]['revision_candidate_exports']


def test_parent_version_changes_reopen_only_the_revision_meaning_and_keep_history(tmp_path):
    agent, lab, child, _ = revision_scene(tmp_path)
    first = route(agent, lab, question={'id': child['id']})
    reference = first['parent_revision_question_ref']
    owner = next(q for q in agent.questions if q['id'] == reference['question_id'])
    answer = deepcopy(child['semantic_answer'])
    agent._save('current.py', "def predict(row): return .6 * row['features']['u']\n")
    agent._execute('evaluate', {'model': 'current.py'}, lab)
    register_theories(agent, [theory('current_parent', binary('mul', constant(.6), feature()))])
    second = route(agent, lab, question={'id': child['id']})
    assert second['parent_revision_question_ref']['meaning_version'] == reference['meaning_version'] + 1
    assert owner['meaning_history'][0]['subject_refs'][0]['version'] == 1
    assert owner['subject_refs'][0]['version'] == 2
    assert child['meaning_version'] == 1 and child['semantic_answer'] == answer


@pytest.mark.parametrize('reason', ['completed', 'solved', 'no_use'])
def test_measurement_answer_without_failed_typed_parent_does_not_invent_revision_work(tmp_path, reason):
    from hegel_machine.cognition.question_continuation import parent_revision_obligation
    agent, lab, child, _ = revision_scene(tmp_path)
    answer = {'parent_revision_completed': reason == 'completed'}
    if reason == 'solved':
        agent.evaluated_model['current_mse'] = 0.
    if reason == 'no_use':
        child['use_sites'] = []
    assert parent_revision_obligation(agent, child, answer) is None
    assert not any(q['id'].startswith('parent_revision:') for q in agent.questions)


def test_reused_measurement_use_keeps_its_unfinished_parent_status(tmp_path):
    agent, lab, child, _ = revision_scene(tmp_path)
    agent._save('shared/parent_use.py', "def select_actions(*args): return {'stop':'No new plan.'}\n")
    child['use_sites'] = [{'kind': 'choose_parent_experiment', 'parent_id': 'task',
                          'path': 'shared/parent_use.py'}]
    applied = agent._execute('run', {'operator': 'research_step',
                                    'research_question': {'id': child['id']}}, lab)
    assert applied['status'] == 'answer_applied' and applied['parent_revision_completed'] is False
    previous = len(child['parent_use_history'])
    routed = route(agent, lab, question={'id': child['id']})
    assert routed['parent_revision_question_ref']['question_id'] != child['id']
    assert len(child['parent_use_history']) == previous


def test_saved_delay_parent_still_matches_the_five_actual_frozen_objects_without_predictions(monkeypatch):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import SearchBudget
    from hegel_machine.cognition.revision_relations import _match_parent
    source = Path('/mnt/hegel-snapshot-cow/pr3_delay_event19_preparation_20261023/parent_contract.json')
    if not source.is_file():
        pytest.skip('The retained D3 event19 structure witness is not mounted')
    contract = json.loads(source.read_text())
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('No historical prediction or fitting'))
    documents = {}
    matched = []
    for retained in contract['matching_objects']:
        reference = retained['candidate_ref']
        if reference['file'] not in documents:
            documents[reference['file']] = json.loads(Path(reference['file']).read_text())
        candidate = documents[reference['file']]
        for key in reference['json_path']:
            candidate = candidate[key]
        bindings, _ = _match_parent(Program.from_dict(candidate['before_pattern']),
            Program.from_dict(contract['program']), candidate['holes'], SearchBudget(10000))
        assert sorted(bindings) == retained['bound_holes']
        matched.append(candidate['id'])
    assert matched == [f'consolidated:8:relation:revision:{i}' for i in (0, 2, 4, 6, 8)]
