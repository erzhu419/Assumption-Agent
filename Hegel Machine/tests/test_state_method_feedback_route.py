"""Paid state-method outcomes return through Common without more world research."""

from copy import deepcopy

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.models import Observation
from scripts import run_state_method_pairing as study
from test_state_method_pairing import paid_fixture


def test_actual_pair_reaches_common_method_question_without_refitting(tmp_path, monkeypatch):
    paid, case, values = paid_fixture(tmp_path)
    registration = study.prepare(paid, tmp_path / 'prepared', case_ref=case,
        evaluation_seed=491, research_compute_cap=10000, compute_cap=30000,
        evaluation_rows=20, approval_ref=paid)
    directory = tmp_path / 'run'

    class Evaluator:
        def __init__(self, **_):
            manifest = read_json_checkpoint(directory / 'frozen_outputs.json')
            assert set(manifest['models']) == {'00', '10'}
            assert not manifest['evaluation_labels_opened']

        def evaluation_rows(self, *, length, seed):
            assert (length, seed) == (20, 491)
            return [Observation(row['features'], row['target'], tuple(row['history']))
                    for record in values['records'] for row in record['rows']]

    monkeypatch.setattr(study, 'AnonymousHistoryLab', Evaluator)
    execute = CommonArchiveInquiryAgent._execute
    routed = []

    def observe_route(agent, action, arguments, lab):
        is_method = (action == 'run' and arguments.get('operator') == 'research_route'
            and arguments.get('research_question', {}).get('kind') == 'method_use')
        if not is_method:
            return execute(agent, action, arguments, lab)
        before = deepcopy(agent.costs)
        observations = deepcopy(agent.observations)
        request = deepcopy(arguments['method_action_revision'])
        tested = study.read(request['tested_candidate_ref'])
        body = deepcopy(agent.theory_consolidation['candidates'][tested['id']])
        assert request['segments']
        assert len(request['pair_refs']) == 1
        assert request['paired_candidate_id'] == tested['id']
        assert 'condition' not in tested['selection_plan']
        with monkeypatch.context() as during_feedback:
            def no_world_research(*args, **kwargs):
                raise AssertionError('consuming a paid pair must not refit or acquire world labels')

            during_feedback.setattr(
                'hegel_machine.cognition.state_search.closed_startup_claims', no_world_research)
            during_feedback.setattr(
                'hegel_machine.cognition.state_search.closed_state_claims', no_world_research)
            during_feedback.setattr(study, 'AnonymousHistoryLab', no_world_research)
            result = execute(agent, action, arguments, lab)
        assert agent.observations == observations
        assert agent.costs['observation_cost'] == before['observation_cost']
        assert agent.costs['model_calls'] == before['model_calls'] == 0
        assert agent.theory_consolidation['candidates'][tested['id']] == body
        routed.append((agent, deepcopy(arguments), result, before))
        return result

    monkeypatch.setattr(CommonArchiveInquiryAgent, '_execute', observe_route)
    result = study.run(registration, directory)
    assert len(routed) == 1
    agent, arguments, returned, before = routed[0]
    assert returned['status'] == 'complete_no_supported_action_revision'
    action = returned['method_action_research']
    assert action['proposals']['status'] == 'actual_frozen_body_reused'
    assert action['segments'] and not action['segments'][0]['missing_dependencies']
    selection = action['paired_selection']
    assert selection['preferences'][0]['preference'] == 'baseline'
    assert not selection['general_guard_supported']
    assert action['installed_candidate'] is None
    assert result['ordinary_use']['ordinary_method_retained']
    assert result['ordinary_use']['ordinary_research_route_executed']
    assert not result['ordinary_use']['method_execution_after_feedback']
    assert not result['ordinary_use']['successors_started']
    owner = next(q for q in agent.questions if q['id'] == registration['question']['id'])
    child = next(q for q in agent.questions
                 if q['id'] == arguments['research_question']['id'])
    assert child['parent_id'] == owner['id']
    assert child['method_use_plan']['candidate_id'] == registration['method_ref']['id']
    checkpoint = read_json_checkpoint(agent.workspace / 'training.json')
    assert checkpoint['costs'] == agent.costs
    assert checkpoint['status'] == 'method_evidence_consumed'
    assert 0 < result['ordinary_use']['new_compute_cost'] <= registration['method_condition_compute_cap']
    assert result['ordinary_use']['new_compute_cost'] == agent.costs['compute_cost'] - before['compute_cost']
    assert study.read(result['ordinary_use']['selection_ref']) == selection


def test_no_completed_research_segment_does_not_create_method_support(tmp_path):
    paid, case, _ = paid_fixture(tmp_path)
    registration = study.prepare(paid, tmp_path / 'prepared', case_ref=case,
        evaluation_seed=492, research_compute_cap=10000, compute_cap=30000,
        evaluation_rows=20, approval_ref=paid)
    directory = tmp_path / 'run'
    directory.mkdir()
    factories = study.make_factories(registration, directory)
    scene = factories['restore_cell']('10', registration['cells']['10'], directory / '10')
    agent = scene['agent']
    before = deepcopy(agent.costs)
    outcome = study.save(directory / '10' / 'outcome.json', {
        'method_artifact_ref': scene['method_artifact_ref']})
    result = factories['ordinary_use']([], {'10': outcome}, directory)
    returned = study.read(result['research_route_ref'])['method_action_research']
    assert returned['status'] == 'method_actions_unresolved'
    assert returned['reason'] == 'actual_method_inputs_incomplete'
    assert returned['segments'] == []
    assert not result['local_program_supported']
    assert not result['general_guard_supported']
    assert result['installed_candidate_id'] is None
    assert result['new_compute_cost'] == agent.costs['compute_cost'] - before['compute_cost'] == 1
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
    assert read_json_checkpoint(agent.workspace / 'training.json')['costs'] == agent.costs
