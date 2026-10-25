"""An actually retained revision export remains the same executable parent."""

from copy import deepcopy

from hegel_machine.cognition.temporal_paired_method import execute_procedure
from test_finite_native_author import POLICY
from test_revision_candidate_exports import prepared_revision_scene


def completed_revision_comparison(tmp_path, monkeypatch):
    agent, lab, question = prepared_revision_scene(tmp_path, monkeypatch)
    agent._before_model_turn(lab)
    expected = [*agent.knowledge['finite_native_author']['predictor_paths'], *agent.candidate_paths]
    for round_ in question['native_condition_research']:
        expected.extend(item['model'] for item in round_['candidates'])
        expected.extend(item['model'] for item in round_['exported_models'])
    expected = list(dict.fromkeys(expected))
    before_compute = agent.costs['compute_cost']
    compared = execute_procedure(agent, lab, POLICY, max_steps=1, allow_confirmation=False,
        record_events=True, before_step=agent._before_model_turn, after_step=agent._after_model_turn)
    assert compared['steps'][0]['proposal']['arguments']['operator'] == 'research_choice'
    feedback = compared['steps'][0]['feedback']
    assert compared['steps'][0]['proposal']['arguments']['models'] == expected
    assert [item['path'] for item in feedback['evaluations']] == expected
    # The current 300-unit construction yields five revision exports. Together
    # with the old archive and two native drafts, all eight are scored on four
    # paid rows; this comparison has no cached scores or extra incumbent pass.
    assert len(expected) == feedback['new_scored_count'] == 8
    assert feedback['paid_rows'] == 4 and feedback['reused_score_count'] == 0
    assert feedback['retained_comparison_score_counts'] == {'reused': 0, 'newly_scored': 0}
    execution_costs = [item['paid_score_reuse']['execution_costs']['compute_cost']
                       for item in feedback['evaluations']]
    assert execution_costs == [4] * 8
    assert feedback['compute_cost'] == sum(execution_costs) == 32
    assert agent.costs['compute_cost'] - before_compute == 32
    retained = agent.evaluated_model
    exported, = [item for item in question['revision_candidate_exports']
                 if item['model'] == retained['source_model']]
    assert retained['current_mse'] == 0. and retained['evaluated_rows'] == 4
    return agent, lab, question, exported


def test_actual_registered_revision_winner_is_the_next_parent_without_new_support(tmp_path, monkeypatch):
    agent, lab, question, exported = completed_revision_comparison(tmp_path, monkeypatch)
    identifier = exported['theory_ref']['id']
    original = deepcopy(agent.theory_index[identifier])
    transfers = deepcopy(agent.theory_consolidation['transfers'])
    costs = deepcopy(agent.costs)
    agent._active_question_id = question['id']
    agent._before_model_turn(lab)
    target = question['theory_target']
    assert target['parent_theory_id'] == identifier
    assert target['parent_binding_role'] == 'retained_revision_program'
    assert target['parent_selection_ref'] == {'file': 'research_state.json', 'json_path': ['evaluated_model']}
    assert 'parent_binding_obligation' not in target
    assert len(target['evidence_checks']) == 4 and all('observation' not in check
                                                   for check in target['evidence_checks'])
    assert agent.theory_index[identifier] == original
    assert agent.theory_consolidation['transfers'] == transfers
    assert agent.costs == costs and agent.costs['model_calls'] == 0 and not agent.confirmations


def test_source_write_does_not_change_the_actual_retained_program_parent(tmp_path, monkeypatch):
    agent, lab, question, exported = completed_revision_comparison(tmp_path, monkeypatch)
    # Ordinary write is allowed to replace an existing predictor. The retained
    # execution snapshot still contains the Program that actually won.
    agent._execute('write', {'path': exported['model'], 'code': 'def predict(row): return 99.\n'}, lab)
    before = deepcopy(agent.costs)
    agent._bind_revision_target(question)
    target = question['theory_target']
    assert target['parent_theory_id'] == exported['theory_ref']['id']
    assert target['parent_binding_role'] == 'retained_revision_program'
    assert agent.evaluated_model['source_model'] == exported['model']
    assert (agent.workspace / agent._evaluated_summary()['model']).read_text() != (
        agent.workspace / exported['model']).read_text()
    assert agent.costs == before
