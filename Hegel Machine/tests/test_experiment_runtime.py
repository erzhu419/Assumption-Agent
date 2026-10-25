"""The frozen runtime owns CPU execution and timing for both study entries."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import weakref
from multiprocessing import get_context

import pytest

from hegel_machine.cognition import checkpoint_storage, experiment_runtime
from scripts import prepare_pr3_source_successor, run_anonymous_revision_inquiry


def test_parallel_trials_share_checkpoint_slot_without_serializing_research():
    if len(experiment_runtime.physical_worker_cpus()) < 2:
        pytest.skip('two physical cores are required for two native workers')
    context = get_context('fork')
    first_saving, second_ready = context.Event(), context.Event()

    def train(job):
        if job == 1:
            assert first_saving.wait(15)
            # This worker is running while the other saves, but may not
            # construct a second large checkpoint at the same time.
            acquired = experiment_runtime._CHECKPOINT_LOCK.acquire(False)
            if acquired:
                experiment_runtime._CHECKPOINT_LOCK.release()
            second_ready.set()
            assert not acquired
        with experiment_runtime.checkpoint_execution():
            if job == 0:
                first_saving.set()
                assert second_ready.wait(15)
        return {'job': job}

    assert experiment_runtime.run_process_trials(train, [0, 1], 2) == [
        {'job': 0}, {'job': 1}]
    assert experiment_runtime._CHECKPOINT_LOCK is None


def test_checkpoint_releases_idle_prediction_pool_and_preserves_cpu_assignment():
    from hegel_machine.cognition import parallel_prediction as predictions
    from hegel_machine.cognition.language import Program

    cpus = tuple(experiment_runtime.physical_worker_cpus()[:2])
    predictions.configure_prediction_workers(cpus)
    try:
        executor = predictions._executor((None,))
        assert executor.submit(predictions._predict_block,
            (Program('const', value=2.), 0, 1)).result(timeout=15) == [2.]
        with experiment_runtime.checkpoint_execution():
            assert predictions._EXECUTOR is None
            assert predictions._CPUS == cpus
    finally:
        predictions.configure_prediction_workers(())


def test_trial_parent_releases_inherited_inputs_after_fork():
    class Payload:
        pass

    inherited = {0: Payload()}
    reference = weakref.ref(inherited[0])

    def train(job):
        assert inherited[job] is not None
        return {'job': job}

    assert experiment_runtime.run_process_trials(train, [0], 1,
        release_parent=inherited.clear) == [{'job': 0}]
    assert inherited == {} and reference() is None


def test_finished_worker_exits_while_other_group_is_still_running():
    if len(experiment_runtime.physical_worker_cpus()) < 2:
        pytest.skip('two physical cores are required for two native workers')
    context = get_context('fork')
    fast_pid = context.SimpleQueue()

    def train(job):
        if job == 1:
            fast_pid.put(os.getpid())
            return {'job': job}
        pid = fast_pid.get()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return {'job': job}
            time.sleep(.01)
        raise AssertionError('the finished group retained its worker until its peer completed')

    try:
        # The second input completes first; cleanup must follow completion,
        # while returned results retain the registered input order.
        assert experiment_runtime.run_process_trials(train, [0, 1], 2) == [
            {'job': 0}, {'job': 1}]
    finally:
        fast_pid.close()


def test_multiple_trials_preserve_order_and_one_cpu_group_per_worker():
    groups = experiment_runtime.physical_cpu_groups(5, 2)

    def train(job):
        return {'job': job, 'cpus': sorted(os.sched_getaffinity(0)), 'pid': os.getpid()}

    result = experiment_runtime.run_process_trials(train, list(range(5)), 2)
    assert [item['job'] for item in result] == list(range(5))
    assert [item['cpus'] for item in result] == [list(groups[index % len(groups)])
                                               for index in range(5)]
    assert len({item['pid'] for item in result}) == len(groups)


def test_registered_entries_import_the_shared_runtime_directly():
    assert run_anonymous_revision_inquiry._native_process_trials is experiment_runtime.run_process_trials
    assert run_anonymous_revision_inquiry._native_worker_cpu_groups is experiment_runtime.physical_cpu_groups
    assert run_anonymous_revision_inquiry._native_worker_cpus is experiment_runtime.physical_worker_cpus
    assert prepare_pr3_source_successor.write_json_checkpoint is checkpoint_storage.write_json_checkpoint


def test_execution_timing_records_failed_phases_without_changing_scientific_costs():
    costs = {'compute_cost': 17, 'python_execution_seconds': 3.5}
    timing = experiment_runtime.ExecutionTiming(('restore', 'author'))
    with pytest.raises(ValueError, match='retained failure'):
        with timing.phase('restore'):
            raise ValueError('retained failure')
    result = timing.result()
    assert result['phase_seconds']['restore'] >= 0
    assert result['phase_seconds']['author'] == 0
    assert result['wall_seconds'] >= result['phase_seconds']['restore']
    assert costs == {'compute_cost': 17, 'python_execution_seconds': 3.5}


def test_new_frozen_checkout_imports_and_executes_runtime_without_live_scripts(tmp_path):
    root = prepare_pr3_source_successor.ROOT
    frozen = tmp_path / 'source_at_run'
    for source in (root / 'src/hegel_machine').rglob('*.py'):
        destination = frozen / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in ('run_anonymous_revision_inquiry', 'prepare_pr3_source_successor',
                 'run_online_inquiry_llm', 'run_online_inquiry_study'):
        destination = frozen / 'scripts' / (name + '.py')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / 'scripts' / destination.name, destination)
    program = """
import os, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root / 'src'), str(root)]
from hegel_machine.cognition import experiment_runtime as runtime
from hegel_machine.cognition import checkpoint_storage as storage
from scripts import prepare_pr3_source_successor as preparation
from scripts import run_anonymous_revision_inquiry as study
assert Path(runtime.__file__).is_relative_to(root)
assert preparation.ROOT == root and study.ROOT == root
assert study._native_process_trials is runtime.run_process_trials
assert study._native_worker_cpu_groups is runtime.physical_cpu_groups
assert preparation.write_json_checkpoint is storage.write_json_checkpoint
parent = os.getpid()
result = runtime.run_process_trials(lambda job: {'job': job, 'pid': os.getpid()}, [0], workers=1)
assert result[0]['job'] == 0 and result[0]['pid'] != parent
print('frozen_runtime_import_and_process_passed')
"""
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    completed = subprocess.run([sys.executable, '-c', program, str(frozen)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == 'frozen_runtime_import_and_process_passed'
