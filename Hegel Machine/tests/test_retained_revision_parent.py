"""An actually retained revision export remains the same executable parent."""

from copy import deepcopy
from pathlib import Path

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from test_finite_native_author import POLICY
from test_revision_candidate_exports import prepared_revision_scene


def completed_revision_comparison(tmp_path, monkeypatch):
    agent, lab, question = prepared_revision_scene(tmp_path, monkeypatch)
    agent._before_model_turn(lab)
    first_export, = question['revision_candidate_exports']
    directory = Path(first_export['model']).parent
    revision_paths = [str(directory / f'candidate_{index:04d}_v0001.py') for index in range(2)]
    assert agent.candidate_paths == revision_paths[:1]
    expected = [*agent.knowledge['finite_native_author']['predictor_paths'], *revision_paths]
    for round_ in question['native_condition_research']:
        expected.extend(item['model'] for item in round_['candidates'])
        expected.extend(item['model'] for item in round_['exported_models'])
    expected = list(dict.fromkeys(expected))
    before_compute = agent.costs['compute_cost']
    construction_cost = []
    def before_step(laboratory):
        before = agent.costs['compute_cost']
        result = agent._before_model_turn(laboratory)
        construction_cost.append(agent.costs['compute_cost'] - before)
        # The next funded binding exports a second actual Program. Keep it in
        # the full comparison even though its subsequent paid MSE is worse.
        assert agent.candidate_paths == revision_paths
        assert question['revision_candidate_exports'][0] == first_export
        for exported in question['revision_candidate_exports']:
            theory = agent.theory_index[exported['theory_ref']['id']]
            assert theory['version'] == exported['theory_ref']['version']
            assert (agent.workspace / exported['model']).read_text() == export_temporal_program(
                Program.from_dict(theory['program']), feature_names=tuple(theory['role_signature']))
        return result
    compared = execute_procedure(agent, lab, POLICY, max_steps=1, allow_confirmation=False,
        record_events=True, before_step=before_step, after_step=agent._after_model_turn)
    assert compared['steps'][0]['proposal']['arguments']['operator'] == 'research_choice'
    feedback = compared['steps'][0]['feedback']
    assert compared['steps'][0]['proposal']['arguments']['models'] == expected
    assert [item['path'] for item in feedback['evaluations']] == expected
    # Both funded revision bindings, the old archive and two native drafts are
    # scored on four paid rows. Construction remains inside its original300 cap.
    assert len(expected) == feedback['new_scored_count'] == 5
    assert feedback['paid_rows'] == 4 and feedback['reused_score_count'] == 0
    assert feedback['retained_comparison_score_counts'] == {'reused': 0, 'newly_scored': 0}
    execution_costs = [item['paid_score_reuse']['execution_costs']['compute_cost']
                       for item in feedback['evaluations']]
    assert execution_costs == [4] * 5
    assert feedback['compute_cost'] == sum(execution_costs) == 20
    assert construction_cost == [110]
    assert agent.costs['compute_cost'] - before_compute == 130
    assert question['mission']['resource_usage']['compute'] == 233 <= 300
    assert next(item for item in feedback['evaluations'] if item['path'] == revision_paths[1])['mse'] == 6.5
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
    agent._bind_revision_target(question)
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
    # The complete hook also performs funded research, separate from the
    # zero-compute parent binding. Its remaining preparation is not free.
    agent._before_model_turn(lab)
    assert agent.costs['compute_cost'] - costs['compute_cost'] == 26
    assert question['mission']['resource_usage']['compute'] == 259 <= 300
    assert agent.theory_index[identifier] == original
    assert agent.theory_consolidation['transfers'] == transfers


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
