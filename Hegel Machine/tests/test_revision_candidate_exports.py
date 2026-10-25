"""Joint revision Programs enter the unchanged ordinary strong Python comparison."""
from copy import deepcopy

from hegel_machine.cognition.finite_native_author import export_finite_native_author, export_revision_candidates
from hegel_machine.cognition import finite_native_author
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from hegel_machine.cognition.theory_consolidation import _read_library, _anchor_file_references
from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent
from test_finite_native_author import ChoiceLab, POLICY
from test_inquiry_revision import action
from test_native_condition_inquiry import PaidConditionalLab
from test_registered_revision_target import revision_scene
from test_revision_context_fitting import FITTING


def prepared_revision_scene(tmp_path, monkeypatch):
    agent, lab, owner = revision_scene(tmp_path)
    bank = owner['native_condition_research'][0]['candidates']
    agent._execute('query', {'actions': [action(-2., -1., 2.)]}, lab)
    # A real zero-budget native continuation has already consumed the new
    # record's metadata. Fixed old bank cells remain ordinary comparison inputs.
    agent._execute('run', {'operator': 'native_condition_research',
        'research_question': {'id': owner['id']}, 'max_compute': 0, 'max_candidates': 2}, lab)
    agent._execute('evaluate', {'model': bank[-1]['model']}, lab)
    owner['theory_target']['revision_fitting'] = {**FITTING, 'max_structures': 8}
    # Existing theory funding bounds construction; the original task account
    # funds ordinary scoring. No extra task allowance is created.
    owner['mission'] = {'purpose': 'theory_learning', 'mission_id': 'fixture_revision_exports',
                        'theory_learning_allocation': {'compute': 300}}
    agent._active_question_id = owner['id']
    agent.knowledge['finite_native_author']['question']['id'] = owner['id']
    agent.knowledge['finite_native_author']['question']['theory_target']['revision_fitting'] = deepcopy(
        owner['theory_target']['revision_fitting'])
    # The stored program freezes this exact current procedure, as in the runner.
    agent._save(POLICY, export_finite_native_author())
    def no_acquisition(*args, **kwargs):
        raise AssertionError('Candidate export/comparison must not buy further fixture data')
    monkeypatch.setattr(lab, 'execute', no_acquisition)
    return agent, lab, owner


def test_registered_revision_exports_all_real_statuses_into_the_full_ordinary_comparison(tmp_path, monkeypatch):
    agent, lab, owner = prepared_revision_scene(tmp_path, monkeypatch)
    before = deepcopy(agent.costs)
    agent._before_model_turn(lab)
    exports = owner.get('revision_candidate_exports', [])
    assert exports, owner.get('frozen_recall_execution', {}).get('result')
    assert {'UNKNOWN', 'candidate_registered'} <= {item['status'] for item in exports}
    original_theories = {item['theory_ref']['id']: deepcopy(agent.theory_index[item['theory_ref']['id']])
                         for item in exports}
    original_transfers = deepcopy(agent.theory_consolidation['transfers'])
    for item in exports:
        actual_transfer = next(record for record in original_transfers
                               if record['theory_ref'] == item['theory_ref'])
        assert item['status'] == actual_transfer['status']
        assert item['support_status'] == original_theories[item['theory_ref']['id']]['support']['status']
    assert all('program' not in item and 'observation' not in item for item in exports)
    for item in exports:
        text = (agent.workspace / item['model']).read_text()
        assert 'def fit(' not in text
        assert item['model'] in agent.candidate_paths
        assert not item['support_updated'] and not item['independent_confirmation']
    export_cost = deepcopy(agent.costs)
    cached = export_revision_candidates(agent, owner)
    assert not cached['changed'] and cached['new_exports'] == 0 and agent.costs == export_cost
    agent._before_model_turn(lab)
    assert agent.costs == export_cost
    outcome = execute_procedure(agent, lab, POLICY, max_steps=1, allow_confirmation=False,
        record_events=True, before_step=agent._before_model_turn, after_step=agent._after_model_turn)
    proposal = outcome['steps'][0]['proposal']
    assert proposal['arguments']['operator'] == 'research_choice', outcome
    expected = [*agent.knowledge['finite_native_author']['predictor_paths'], *agent.candidate_paths]
    for round_ in owner['native_condition_research']:
        expected.extend(item['model'] for item in round_['candidates'])
        expected.extend(item['model'] for item in round_['exported_models'])
    expected = list(dict.fromkeys(expected))
    assert proposal['arguments']['models'] == expected
    choice = agent.operator_results[-1]
    assert choice['operator'] == 'research_choice'
    assert [item['path'] for item in choice['evaluations']] == expected
    assert choice['paid_rows'] == 4 and choice['compute_cost'] == 4 * len(expected)
    assert {'archive/failed.py', *(item['model'] for item in exports)} <= set(expected)
    assert agent.theory_consolidation['transfers'][:len(original_transfers)] == original_transfers
    evaluations = {item['path']: item for item in choice['evaluations']}
    for item in exports:
        identifier = item['theory_ref']['id']
        original, current = original_theories[identifier], agent.theory_index[identifier]
        assert _anchor_file_references({key: value for key, value in current.items()
            if key not in {'support', 'compatible'}}, agent.workspace) == _anchor_file_references({
            key: value for key, value in original.items() if key not in {'support', 'compatible'}}, agent.workspace)
        if evaluations[item['model']]['mse'] > agent.config.confirmation_mse_threshold:
            assert current['support']['status'] == 'refuted' and not current['compatible'], [
                item['reason'] for item in outcome['steps'][0]['feedback'][
                    'revision_comparison_failures']['unknown']]
        else:
            assert current == original
    assert agent.costs['compute_cost'] <= agent.config.compute_budget
    assert agent.costs['observation_cost'] == before['observation_cost'] == 4
    assert agent.costs['model_calls'] == 0 and not agent.confirmations and agent.model is None


def test_restored_relative_index_bindings_match_the_anchored_actual_transfer(tmp_path, monkeypatch):
    agent, lab, owner = prepared_revision_scene(tmp_path, monkeypatch)
    # Save the actual generated Programs before the newly introduced export
    # hook, matching the supported older checkpoint's physical representation.
    with monkeypatch.context() as patch:
        patch.setattr(finite_native_author, 'export_revision_candidates',
                      lambda *_: {'changed': False, 'new_exports': 0})
        agent._before_model_turn(lab)
    assert not owner.get('revision_candidate_exports')
    agent._write_evidence()
    agent._prepare_program_state()
    previous = agent.result('researching')
    restored = RegisteredOrganizationAgent(None, tmp_path / 'restored', arm='fixed_pr2',
        research_organization=deepcopy(agent._organization_registration), source_task=False,
        knowledge={'files': {}}, config=agent.config)

    class RestorableChoiceLab(ChoiceLab):
        costs = PaidConditionalLab.costs
        restore_paid_account = PaidConditionalLab.restore_paid_account

    restored_lab = RestorableChoiceLab()
    restored.restore_research(previous, agent.workspace, restored_lab, confirmation_seed=0)
    # Hydrating the physical library uses its existing reader; the restored
    # local index intentionally still retains its original relative references.
    restored.theory_consolidation = _read_library(
        {'file': str(restored.workspace / 'shared/theory_library.json'), 'json_path': []},
        restored.workspace)
    current = next(q for q in restored.questions if q['id'] == owner['id'])
    transfer = restored.theory_consolidation['transfers'][0]
    binding = restored.theory_index[transfer['theory_id']]['consolidation_ref']
    assert binding['fitting']['evidence_refs'][0]['file'] == 'laboratory_observations.json'
    assert transfer['binding']['fitting']['evidence_refs'][0]['file'] == str(
        restored.workspace / 'laboratory_observations.json')
    before = deepcopy(restored.costs)
    exported = export_revision_candidates(restored, current)
    assert exported['new_exports'] == len(restored.theory_consolidation['transfers'])
    assert not exported['errors']
    assert {item['status'] for item in exported['exports']} == {'candidate_registered', 'UNKNOWN'}
    for item in exported['exports']:
        original = next(record for record in restored.theory_consolidation['transfers']
                        if record['theory_ref'] == item['theory_ref'])
        assert item['status'] == original['status']
    assert restored.costs['compute_cost'] == before['compute_cost']
    assert restored.costs['observation_cost'] == before['observation_cost'] == 4
    after = deepcopy(restored.costs)
    assert not export_revision_candidates(restored, current)['changed']
    assert restored.costs == after
