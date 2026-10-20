"""The next author receives completed revision outcomes without a prescribed policy."""

from copy import deepcopy
import json
from types import SimpleNamespace

from scripts import run_native_method_compilation as compilation
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH


def test_completed_revision_comparison_keeps_same_case_costs_and_source_paths(tmp_path):
    reference = tmp_path / 'frozen'
    reference.mkdir()
    (reference / 'published_result.json').write_text(json.dumps({'method_source': 'old'}))
    workspace = tmp_path / 'author'
    (workspace / 'shared').mkdir(parents=True)
    (workspace / METHOD_PATH).write_text('new')
    trials = [
        {'case': 'same_interface_a', 'directory': 'trials/trial_000', 'method_source': 'old',
         'status': 'completed', 'costs': {'observation_cost': 552, 'python_execution_seconds': 100.},
         'fresh_confirmation': {'passed': True},
         'execution_summary': {'action_counts': {'query': 3, 'model': 1}}},
        {'case': 'same_interface_a', 'directory': 'trials/trial_001', 'method_source': 'new',
         'status': 'completed', 'costs': {'observation_cost': 808, 'python_execution_seconds': 4.},
         'fresh_confirmation': {'passed': True},
         'execution_summary': {'action_counts': {'query': 5, 'model': 1}}},
        {'case': 'same_interface_b', 'directory': 'trials/trial_002', 'method_source': 'old',
         'status': 'method_stopped', 'fresh_confirmation': None},
        {'case': 'same_interface_b', 'directory': 'trials/trial_003', 'method_source': 'new',
         'status': 'completed', 'fresh_confirmation': {'passed': True}},
    ]
    saved = deepcopy(trials)
    author = SimpleNamespace(workspace=workspace, files={}, common_interface='', practice_trials=trials)
    plan = {'repair_development_cases': [], 'published_revision': {'frozen_study': str(reference)}}
    compilation.attach_transfer_repair_experience(author, plan)
    evidence = json.loads(author.files['repair_evidence.json'])
    comparison = evidence['completed_revision_comparison']
    assert not comparison['source_unchanged']
    first, second = comparison['pairs']
    assert first['frozen_parent']['costs']['observation_cost'] == 552
    assert first['current']['costs']['observation_cost'] == 808
    assert first['current']['action_counts'] == {'query': 5, 'model': 1}
    assert first['frozen_parent']['source_path'] == 'trials/trial_000/' + METHOD_PATH
    assert second['frozen_parent']['status'] == 'method_stopped'
    assert second['frozen_parent']['action_counts'] is None
    assert 'costs' not in second['frozen_parent']
    assert 'selected' not in comparison
    assert 'completed_revision_comparison:' in author.common_interface
    assert '"observation_cost":552' in author.common_interface
    assert '"observation_cost":808' in author.common_interface
    assert 'action and arguments are top-level' in author.common_interface
    assert 'rows is an integer count' in author.common_interface
    assert 'relative to that counterexample.json directory' in author.common_interface
    assert 'feedback.stdout_file' in author.common_interface
    assert author.practice_trials == saved
    assert (workspace / METHOD_PATH).read_text() == 'new'
    assert not (workspace / 'trials').exists()


def test_unchanged_source_does_not_duplicate_or_invent_a_revision_comparison(tmp_path):
    (tmp_path / 'published_result.json').write_text(json.dumps({'method_source': 'same'}))
    (tmp_path / 'shared').mkdir()
    (tmp_path / METHOD_PATH).write_text('same')
    author = SimpleNamespace(workspace=tmp_path, files={}, common_interface='', practice_trials=[])
    compilation.attach_transfer_repair_experience(author, {
        'repair_development_cases': [], 'published_revision': {'frozen_study': str(tmp_path)}})
    comparison = json.loads(author.files['repair_evidence.json'])['completed_revision_comparison']
    assert comparison['source_unchanged'] and comparison['pairs'] == []
