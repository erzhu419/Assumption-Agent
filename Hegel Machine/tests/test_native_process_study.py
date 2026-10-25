"""Native CPU execution preserves the existing study and intervention contracts."""

from copy import deepcopy
import json
from multiprocessing import current_process, get_context
import os
from pathlib import Path

import pytest

from scripts import run_anonymous_revision_inquiry as study
from hegel_machine.cognition import experiment_runtime
from test_registered_native_author import (
    SOURCE, LocalLab, knowledge, paid_prefix, plan, runtime_root,
)
from test_registered_target_intervention import effects, frozen_pair, removed_plan


def case_plan(steps):
    configured = plan(steps=steps)
    first = configured['cases'][0]
    configured['cases'] = [{**first, 'id': f'case_{index:03}'} for index in range(3)]
    return configured


def job_keys(configured):
    arms = configured['arms']
    return [(case['id'], arm) for i, case in enumerate(configured['cases'])
            for arm in arms[i % len(arms):] + arms[:i % len(arms)]]


@pytest.mark.parametrize(('jobs', 'workers', 'widths'), [
    (2, 8, [4, 4]), (8, 8, [1] * 8), (9, 8, [1] * 8),
    (3, 8, [3, 3, 2]), (2, 1, [1]), (1, 8, [8]),
])
def test_native_core_groups_share_one_global_cap(monkeypatch, jobs, workers, widths):
    monkeypatch.setattr(experiment_runtime, 'physical_worker_cpus', lambda: list(range(12)))
    groups = study._native_worker_cpu_groups(jobs, workers)
    assert [len(group) for group in groups] == widths
    assigned = [cpu for group in groups for cpu in group]
    assert assigned == list(range(min(workers, 12)))
    assert len(assigned) == len(set(assigned))


def test_two_native_coordinators_have_disjoint_four_core_groups_and_can_start_children(tmp_path):
    if len(study._native_worker_cpus()) < 8:
        pytest.skip('This execution check requires eight physical cores')
    barrier = get_context('fork').Barrier(2, timeout=20)

    def train(index):
        workspace = tmp_path / str(index)
        workspace.mkdir()
        barrier.wait()
        receipt = {'pid': os.getpid(), 'daemon': current_process().daemon,
                   'cpus': sorted(os.sched_getaffinity(0))}
        child_file = workspace / 'child.json'

        def child():
            child_file.write_text(json.dumps({'pid': os.getpid(),
                'cpus': sorted(os.sched_getaffinity(0))}))

        nested = get_context('fork').Process(target=child)
        nested.start()
        nested.join(20)
        assert nested.exitcode == 0
        result = {'workspace': str(workspace), 'receipt': receipt,
                  'child': json.loads(child_file.read_text())}
        (workspace / 'training.json').write_text(json.dumps(result))
        return {'index': index, 'result': result}

    trials = study._native_process_trials(train, (0, 1), workers=8)
    assert [trial['index'] for trial in trials] == [0, 1]
    first, second = (trial['result']['receipt'] for trial in trials)
    assert first['pid'] != second['pid']
    assert len(first['cpus']) == len(second['cpus']) == 4
    assert not set(first['cpus']) & set(second['cpus'])
    assert set(first['cpus']) | set(second['cpus']) == set(study._native_worker_cpus()[:8])
    for trial in trials:
        result = trial['result']
        assert result['receipt']['daemon'] is False
        assert result['child']['pid'] != result['receipt']['pid']
        assert result['child']['cpus'] == result['receipt']['cpus']


def test_eight_native_processes_use_distinct_physical_cores_and_keep_full_pool(
        tmp_path, runtime_root):
    if len(study._native_worker_cpus()) < 8:
        pytest.skip('This execution check requires the requested eight physical cores')
    configured = case_plan(1)
    configured['limits']['execution_seconds'] = 1800.
    output = tmp_path / 'eight'
    keys = job_keys(configured)
    barrier = get_context('fork').Barrier(8, timeout=20)
    memory = knowledge("import json, os\n"
        "def select_actions(*args):\n"
        "    json.dump({'pid':os.getpid(), 'cpus':sorted(os.sched_getaffinity(0)), "
        "'blas':os.environ['OPENBLAS_NUM_THREADS']}, open('subprocess_worker.json','w'))\n"
        "    return {'stop': 'Retained negative fixture.'}\n")
    pool = {f'archive/old_{index:03}.py': f'# retained old draft {index}\n' for index in range(999)}
    failure = {'status': 'refuted', 'counterexamples': [{'observation_index': 0}],
               'costs': {'compute_cost': 17, 'python_execution_seconds': 3.5}}
    memory['files'].update(pool)
    memory['files']['shared/retained_failure.json'] = json.dumps(failure)
    before_environment = {name: os.environ.get(name) for name in
        ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    parent_pid = os.getpid()

    def factory(*args, **kwargs):
        agent = study.RegisteredOrganizationAgent(*args, **kwargs)
        key = (agent.workspace.parent.name, agent.workspace.name)
        if key[0] == 'case_002':
            agent.costs['python_execution_seconds'] = 1800.5
            agent._prepare_program_state()  # Restored jobs retain their canonical checkpoint.
        if key in keys[:8]:
            # This fails if eight requested workers run serially or in threads.
            barrier.wait()
        receipt = {'pid': os.getpid(), 'cpus': sorted(os.sched_getaffinity(0)),
            'thread_limits': {name: os.environ.get(name) for name in before_environment},
            'execution_seconds': agent.config.execution_seconds}
        (agent.workspace / 'worker.json').write_text(json.dumps(receipt))
        return agent

    summary = study.run_study(configured, output, knowledge=memory,
        prepare_prefix=paid_prefix, laboratory_factory=lambda _: LocalLab(),
        provider_factory=lambda _: pytest.fail('Native work must not construct a provider'),
        agent_factory=factory, workers=8)
    assert [(row['case'], row['arm']) for row in summary['trials']] == keys
    receipts = [json.loads((output / case / arm / 'worker.json').read_text()) for case, arm in keys]
    assert len({item['pid'] for item in receipts}) == 8
    assert parent_pid not in {item['pid'] for item in receipts}
    assert len({tuple(item['cpus']) for item in receipts}) == 8
    physical = []
    for receipt in receipts:
        topology = Path(f"/sys/devices/system/cpu/cpu{receipt['cpus'][0]}/topology")
        physical.append(((topology / 'physical_package_id').read_text(),
                         (topology / 'core_id').read_text()))
    assert len(set(physical)) == 8
    assert all(len(item['cpus']) == 1 and set(item['thread_limits'].values()) == {'1'}
               for item in receipts)
    assert all(item['execution_seconds'] == configured['limits']['execution_seconds'] for item in receipts)
    assert {name: os.environ.get(name) for name in before_environment} == before_environment
    for row in summary['trials']:
        workspace = output / row['case'] / row['arm']
        training = json.loads((workspace / 'training.json').read_text())
        if row['case'] == 'case_002':
            assert row['status'] == 'python_execution_budget_exhausted'
            assert row['costs']['python_execution_seconds'] == 1800.5
            assert not (workspace / 'subprocess_worker.json').exists()
        else:
            child = json.loads((workspace / 'subprocess_worker.json').read_text())
            worker = json.loads((workspace / 'worker.json').read_text())
            assert child['pid'] != worker['pid'] and child['cpus'] == worker['cpus']
            assert child['blas'] == '1'
            assert row['status'] == 'method_stopped'
        assert training['status'] == row['status']
        assert row['model'] is None and row['terminal']['mse'] is None
        assert row['costs'] == training['costs']
        assert row['costs']['model_calls'] == row['costs']['reported_tokens'] == 0
        assert row['costs']['python_execution_seconds'] > 0
        assert all((workspace / path).read_text() == text for path, text in pool.items())
        assert json.loads((workspace / 'shared/retained_failure.json').read_text()) == failure


def test_native_parallel_and_sequential_have_the_same_outcomes_and_terminal_boundary(
        tmp_path, runtime_root):
    configured = case_plan(3)
    driver = SOURCE.replace("    state = json.load(open('research_state.json'))",
        "    from pathlib import Path\n"
        "    if Path.cwd().parent.name == 'case_001':\n"
        "        return {'stop':'Retain this failed fixture without a submitted model.'}\n"
        "    state = json.load(open('research_state.json'))")
    runs = []
    for workers in (1, 8):
        output = tmp_path / f'workers_{workers}'
        paths = [output / case / arm / 'training.json' for case, arm in job_keys(configured)]
        terminal_calls = []

        class GuardedLab(LocalLab):
            def evaluation_rows(self, *args, **kwargs):
                assert all(path.is_file() for path in paths)
                terminal_calls.append(os.getpid())
                return super().evaluation_rows(*args, **kwargs)

        summary = study.run_study(deepcopy(configured), output, knowledge=knowledge(driver),
            prepare_prefix=paid_prefix, laboratory_factory=lambda _: GuardedLab(), workers=workers)
        assert terminal_calls == [os.getpid()] * 6
        compared = []
        for row in summary['trials']:
            training = json.loads((output / row['case'] / row['arm'] / 'training.json').read_text())
            costs = {key: value for key, value in row['costs'].items() if key != 'python_execution_seconds'}
            assert row['costs'] == training['costs']  # Queueing does not add/recharge training work.
            terminal = row['terminal']
            predictions = (json.loads(Path(terminal['prediction_ref']['file']).read_text())['predictions']
                           if 'prediction_ref' in terminal else None)
            compared.append((row['case'], row['arm'], row['status'], row['model'], costs,
                row['revision_count'], row['fresh_confirmation_passed'], terminal['passed'],
                terminal['mse'], terminal['evaluation_rows'], predictions, training['paid_observations']))
        runs.append(compared)
    assert runs[0] == runs[1]


def test_native_process_intervention_executes_only_fresh_C_and_keeps_costs(
        tmp_path, runtime_root, effects):
    configured, keep = frozen_pair(tmp_path)
    selected = removed_plan(tmp_path, configured)
    destination = tmp_path / 'removed_parallel'

    def factory(*args, **kwargs):
        assert kwargs['arm'] == 'pr3'
        assert kwargs['knowledge']['files'] == knowledge()['files']
        agent = study.RegisteredOrganizationAgent(*args, **kwargs)
        assert set(agent.theory_consolidation['candidates']) == {'old', 'new_alias'}
        (agent.workspace / 'worker.json').write_text(json.dumps({'pid': os.getpid()}))
        return agent

    result = study.run_target_intervention(selected, destination, keep_study=keep,
        prepare_prefix=paid_prefix, laboratory_factory=lambda _: LocalLab(),
        agent_factory=factory, workers=8)
    row, = result['trials']
    workspace = destination / 'case_000/pr3'
    assert json.loads((workspace / 'worker.json').read_text())['pid'] != os.getpid()
    assert row['arm'] == 'pr3' and row['terminal']['passed']
    assert row['costs'] == json.loads((workspace / 'training.json').read_text())['costs']
    assert row['costs']['observation_cost'] == 4 and row['costs']['model_calls'] == 0
    assert not (destination / 'case_000/archive').exists()
    assert not (destination / 'case_000/fixed_pr2').exists()
    assert result['target_intervention']['keep_costs_recharged'] is False
    assert len(result['target_intervention']['keep_outcome_refs']) == 3


def test_native_process_recovery_references_existing_training_without_recharging(
        tmp_path, runtime_root, effects):
    configured, keep = frozen_pair(tmp_path)
    recovered = tmp_path / 'recovered_parallel'
    result = study.run_study(configured, recovered, previous_study=keep, workers=8,
        prepare_prefix=lambda *a: pytest.fail('Completed training must not repurchase a prefix'),
        agent_factory=lambda *a, **k: pytest.fail('Completed training must not reconstruct an agent'),
        laboratory_factory=lambda _: LocalLab())
    for row in result['trials']:
        old_path = keep / row['case'] / row['arm'] / 'training.json'
        assert row['costs'] == json.loads(old_path.read_text())['costs']
        column = json.loads(Path(row['terminal']['prediction_ref']['file']).read_text())
        assert column['training_ref']['file'] == str(old_path)
        assert not (recovered / row['case'] / row['arm'] / 'training.json').exists()


def test_native_compact_summaries_never_read_full_training_and_keep_order(tmp_path, monkeypatch):
    if len(study._native_worker_cpus()) < 2:
        pytest.skip('Two physical cores are needed for the actual process route')
    ready = get_context('fork').Barrier(2, timeout=20)
    original_read = Path.read_text

    def forbid_training_read(path, *args, **kwargs):
        if path.name == 'training.json':
            raise AssertionError('compact job must not reparse its stored training')
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', forbid_training_read)

    def train(index):
        workspace = tmp_path / str(index)
        workspace.mkdir()
        ready.wait()
        # Persist a complete outcome just as the saved-scene branch does.
        (workspace / 'training.json').write_text(json.dumps({'history': 'x' * 65536}))
        return {'index': index, 'status': 'negative_preserved',
            'result_ref': {'file': str(workspace / 'training.json'), 'json_path': []},
            'pid': os.getpid(), 'daemon': current_process().daemon,
            'costs': {'compute_cost': 17, 'model_calls': 0}, 'failure': 'retained'}

    trials = study._native_process_trials(train, (0, 1), workers=2)
    assert [trial['index'] for trial in trials] == [0, 1]
    assert len({trial['pid'] for trial in trials}) == 2
    assert all(trial['daemon'] is False and 'result' not in trial for trial in trials)
    assert all(trial['costs'] == {'compute_cost': 17, 'model_calls': 0} for trial in trials)
    assert all(trial['failure'] == 'retained' for trial in trials)


def test_native_process_failure_preserves_finished_and_failed_job_evidence(tmp_path):
    if len(study._native_worker_cpus()) < 2:
        pytest.skip('Two physical cores are needed for the actual process route')

    def train(index):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / 'training.json').write_text(json.dumps({'status': 'failed' if index else 'completed',
            'costs': {'compute_cost': 23}, 'counterexamples': ['preserved']}))
        if index:
            raise ValueError('actual worker failure retained')
        return {'index': index, 'result_ref': {'file': str(folder / 'training.json'), 'json_path': []}}

    with pytest.raises(ValueError, match='actual worker failure retained'):
        study._native_process_trials(train, (0, 1), workers=2)
    for index in (0, 1):
        saved = json.loads((tmp_path / str(index) / 'training.json').read_text())
        assert saved['costs'] == {'compute_cost': 23}
        assert saved['counterexamples'] == ['preserved']


def test_native_workers_limit_already_loaded_blas_and_preserve_parent_limit(tmp_path):
    from threadpoolctl import threadpool_info, threadpool_limits

    if len(study._native_worker_cpus()) < 2:
        pytest.skip('Two physical cores are needed for the actual process route')

    def train(index):
        return {'index': index, 'pid': os.getpid(),
                'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'blas_threads': [pool['num_threads'] for pool in threadpool_info()
                                 if pool['user_api'] == 'blas']}

    with threadpool_limits(limits=4, user_api='blas'):
        before = [pool['num_threads'] for pool in threadpool_info() if pool['user_api'] == 'blas']
        assert before and set(before) == {4}
        result = study._native_process_trials(train, (0, 1), workers=2)
        assert all(row['blas_threads'] and set(row['blas_threads']) == {1} for row in result)
        assert all(len(row['cpu_affinity']) == 1 for row in result)
        assert [pool['num_threads'] for pool in threadpool_info()
                if pool['user_api'] == 'blas'] == before
