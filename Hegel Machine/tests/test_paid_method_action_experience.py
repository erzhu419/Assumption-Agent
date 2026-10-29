"""Method formation reads original witnesses without inventing step rewards."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
from hegel_machine.cognition.method_action_research import _paid_action_experiences, research_method_actions


def saved(directory, name, value):
    path = directory / name
    write_json_checkpoint(path, value)
    return {'file': str(path), 'json_path': []}


def paid_source(tmp_path):
    program = {'op': 'const', 'value': 0., 'children': []}
    scene = saved(tmp_path, 'before.json', {'theory_index': {'old': {'program': program}}})
    profile = saved(tmp_path, 'profile.json', {'research': {'candidate_pool': {
        'status': 'observed', 'witnesses': [{'models': ['old.py', 'old_alias.py'],
            'execution_ref': {'file': 'paid_score.json'}, 'transition': {
                'status': 'observed', 'guard_depth': 1,
                'startup_first_divergences': 2, 'mature_first_divergences': 0}}]}}})
    native = saved(tmp_path, 'native.json', {'parent_theory_ids': ['old', 'other'],
        'candidates': [], 'comparison_completed': True})
    outcome = {'start_state_ref': scene, 'profile_ref': profile,
        'execution_complete': True, 'comparison': {
            'models_requested': 6, 'models_completed': 6, 'truncated': None},
        'costs': {'compute_cost': 100, 'observation_cost': 0}, 'loss': 0.,
        'native_representation_returns': [{'native_result_ref': native}]}
    outcome_ref = saved(tmp_path, 'outcome.json', outcome)
    supplied = {'source_task_ref': {'file': 'original_request.json', 'json_path': ['source_id']},
        'outcome_ref': outcome_ref}
    return supplied, outcome, program


def test_original_pre_use_parent_and_return_are_loaded_without_fabricated_reward(tmp_path):
    source, outcome, program = paid_source(tmp_path)
    before = deepcopy(outcome)
    experiences, excluded = _paid_action_experiences(
        SimpleNamespace(workspace=tmp_path), [source])
    assert not excluded and len(experiences) == 1
    result = experiences[0]
    assert result['parent_programs'] == [{'id': 'old', 'program': program,
        'source_ref': {'file': str(tmp_path / 'before.json'),
                       'json_path': ['theory_index', 'old', 'program']}}]
    assert source['outcome_ref'] in result['source_refs']
    assert result['source_task_ref'] == source['source_task_ref']
    assert 'gain' not in result and 'preference' not in result
    assert outcome == before


def test_incomplete_comparison_cannot_supply_a_completed_formation_example(tmp_path):
    source, outcome, _ = paid_source(tmp_path)
    outcome['comparison']['models_completed'] = 5
    write_json_checkpoint(Path(source['outcome_ref']['file']), outcome)
    experiences, excluded = _paid_action_experiences(
        SimpleNamespace(workspace=tmp_path), [source])
    assert experiences == []
    assert excluded == [{'outcome_ref': source['outcome_ref'],
                         'reason': 'completed_whole_research_comparison_required'}]


def test_unfinished_content_analysis_remains_unresolved(tmp_path, monkeypatch):
    ordinary = saved(tmp_path, 'ordinary.json', {'action': 'run', 'arguments': {}})
    monkeypatch.setattr('hegel_machine.cognition.research_episodes.extract_method_segment',
        lambda *args, **kwargs: {'missing_dependencies': []})
    monkeypatch.setattr('hegel_machine.cognition.finite_native_author.publish_research_options',
        lambda *args, **kwargs: [])
    monkeypatch.setattr('hegel_machine.cognition.finite_native_author.research_segment_options',
        lambda *args, **kwargs: {'options': [], 'excluded': []})
    monkeypatch.setattr('hegel_machine.cognition.research_programs.propose_research_action_revisions',
        lambda *args, **kwargs: {'candidates': [], 'analysis_compute_cost': 3,
            'experience_revision': {'status': 'analysis_budget_unresolved'}})
    agent = SimpleNamespace(workspace=tmp_path, observations=[], knowledge={},
        costs={'compute_cost': 0}, _program_state=lambda: {})
    result = research_method_actions(agent, {}, {'ordinary_proposal_ref': ordinary,
        'ordinary_method_path': 'ordinary.py', 'segments': [{
            'root_result_ref': ordinary, 'decision_start_ref': ordinary}]},
        public_spec={}, compute_limit=5)
    assert result['status'] == 'method_actions_unresolved'
    assert result['reason'] == 'method_content_analysis_not_completed'
    assert result['new_compute_cost'] == agent.costs['compute_cost'] == 5
