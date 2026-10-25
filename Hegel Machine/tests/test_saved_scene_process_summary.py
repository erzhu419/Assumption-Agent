"""Use the actual saved-scene process entry with no research or prediction work."""

from copy import deepcopy
import json
from multiprocessing import current_process, get_context
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_pr3_source_successor as runner
from scripts import run_anonymous_revision_inquiry as study


class SavedAgent:
    def __init__(self, workspace):
        self.workspace = workspace
        self.costs = {'observation_cost': 145, 'compute_cost': 101,
                      'python_execution_seconds': 13., 'model_calls': 0}
        self.candidate_paths = ['archive/old_999.py']
        workspace.mkdir(parents=True)

    def _save(self, name, text):
        path = self.workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def result(self, status):
        return {'workspace': str(self.workspace), 'status': status, 'costs': deepcopy(self.costs),
                'events': [], 'model': None, 'fresh_confirmation': None,
                'evaluated_model': None, 'complete_history': 'h' * 65536}


def test_real_saved_scene_entry_uses_compact_process_contract(tmp_path, monkeypatch):
    if len(study._native_worker_cpus()) < 8:
        pytest.skip('This checks the actual two jobs with four physical cores each')
    source_tree = tmp_path / 'checkout'
    for name in ('src/hegel_machine/__init__.py', 'scripts/run_anonymous_revision_inquiry.py',
                 'scripts/prepare_pr3_source_successor.py', 'scripts/run_online_inquiry_llm.py',
                 'scripts/run_online_inquiry_study.py'):
        path = source_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# source provenance fixture\n')
    monkeypatch.setattr(runner, 'ROOT', source_tree)
    output = tmp_path / 'result'
    carry = {'observation_cost': 145, 'compute_cost': 101,
             'python_execution_seconds': 13., 'model_calls': 0}
    arms = {}
    for name in ('fixed_pr2', 'pr3'):
        agent = SavedAgent(output / name)
        arms[name] = {'agent': agent, 'lab': SimpleNamespace(), 'previous': {},
            'starting_costs': deepcopy(carry),
            'validation': {'acceptance': {'new_costs': {'compute_cost': 1}}}}
    manifest = {'library_views': {'new_ids': ['new_relation']}}
    accepted = {'preparation': tmp_path / 'prepared', 'output': output,
        'manifest': manifest, 'source': tmp_path / 'original',
        'plan': {'author': {'path': 'shared/driver.py', 'max_steps': 8}},
        'case': {'kind': 'delay'}, 'carry': carry, 'limits': SimpleNamespace(confirmation_mse_threshold=1e-10),
        'acceptance': {'status': 'both_arms_accepted_before_scientific_execution'}, 'arms': arms}
    monkeypatch.setattr(runner, '_freeze_saved_scene_libraries', lambda _: {'frozen': True})
    monkeypatch.setattr('hegel_machine.cognition.finite_native_author.export_finite_native_author',
                        lambda: "def select_actions(*args): return {'stop': 'fixture'}\n")
    monkeypatch.setattr('hegel_machine.cognition.method_effect_query.registered_successor_effect',
                        lambda *_args, **_kwargs: {'cases': []})

    ready = get_context('fork').Barrier(2, timeout=20)

    def finished_without_research(agent, _lab, _plan, _checkpoint, **_options):
        ready.wait()
        agent.costs['compute_cost'] += 7
        return {'status': 'method_step_limit', 'native_attempts': 0, 'new_method_results': 0,
                'boundary': None, 'worker_pid': os.getpid(),
                'worker_daemon': current_process().daemon,
                'worker_cpus': sorted(os.sched_getaffinity(0))}

    monkeypatch.setattr(runner, '_source_research_steps', finished_without_research)
    original_read = Path.read_text

    def no_training_reparse(path, *args, **kwargs):
        if path.name == 'training.json':
            raise AssertionError('the saved-scene summary does not consume full training')
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', no_training_reparse)
    result = runner.run_prepared_saved_scene(tmp_path / 'prepared', accepted_scene=accepted)
    assert [row['arm'] for row in result['trials']] == ['fixed_pr2', 'pr3']
    assert result['additional_costs'] == {'observation_cost': 0, 'compute_cost': 14,
                                        'python_execution_seconds': 0., 'model_calls': 0}
    assert result['new_terminal_evaluations'] == result['new_confirmation_rows'] == 0
    first, second = result['trials']
    assert first['worker_pid'] != second['worker_pid']
    assert first['worker_daemon'] is second['worker_daemon'] is False
    assert len(first['worker_cpus']) == len(second['worker_cpus']) == 4
    assert not set(first['worker_cpus']) & set(second['worker_cpus'])
    assert all('result' not in row for row in result['trials'])
    for row in result['trials']:
        persisted = json.loads(original_read(Path(row['result_ref']['file'])))
        assert persisted['complete_history'] == 'h' * 65536
        assert persisted['costs'] == row['costs']
