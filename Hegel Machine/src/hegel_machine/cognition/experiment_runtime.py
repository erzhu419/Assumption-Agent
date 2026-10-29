"""Shared CPU execution and timing for already authorized research jobs.

Scientific budgets, author actions and candidate decisions stay with the caller.
The runtime only assigns physical cores, bounds nested numerical workers and
returns compact retained-file references between coordinating processes.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import gc
import json
from multiprocessing import get_context
import os
from pathlib import Path
import signal
from time import monotonic, process_time

from threadpoolctl import threadpool_limits

from .parallel_prediction import configure_prediction_workers, shutdown_prediction_workers


_TRAIN = None
_BLAS_LIMITS = None
_CHECKPOINT_LOCK = None


class PythonExecutionDeadline(ValueError):
    """A bounded host preparation or score exhausted its Python allowance."""


class ActivityExecutionDeadline(RuntimeError):
    """The whole research activity window ended, including its host work."""


@contextmanager
def python_execution_deadline(seconds, *, activity=False):
    """Bound synchronous preparation/scoring in the worker's main thread.

    Local score limits return partial Python work through their existing caller.
    An activity limit instead ends the whole research unit. Nested score limits
    preserve the earlier parent's exception so they cannot consume that cutoff.
    """
    deadline_error = ActivityExecutionDeadline if activity else PythonExecutionDeadline
    message = ('Research activity time budget exhausted' if activity
               else 'Python execution time budget exhausted')
    if seconds <= 0:
        raise deadline_error(message)
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = monotonic()
    parent_expires_first = bool(previous_timer[0] and previous_timer[0] <= seconds)
    parent_delivered = False

    def exhausted(signum, frame):
        nonlocal parent_delivered
        if parent_expires_first and callable(previous_handler):
            # The enclosing activity still owns its earlier cutoff. A local
            # scoring timer must not relabel it as a catchable Python limit.
            parent_delivered = True
            previous_handler(signum, frame)
        raise deadline_error(message)

    signal.signal(signal.SIGALRM, exhausted)
    signal.setitimer(signal.ITIMER_REAL,
        min(seconds, previous_timer[0]) if previous_timer[0] else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] and not parent_delivered:
            signal.setitimer(signal.ITIMER_REAL,
                max(.000001, previous_timer[0] - (monotonic() - started)), previous_timer[1])


def physical_worker_cpus():
    """Select one permitted logical CPU from each physical core on this host."""
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical = ((topology / 'physical_package_id').read_text().strip(),
                    (topology / 'core_id').read_text().strip())
        cores.setdefault(physical, cpu)
    return list(cores.values())


def physical_cpu_groups(job_count, workers):
    """Share the existing global physical-core allowance across active jobs."""
    available = physical_worker_cpus()[:workers]
    count = min(job_count, len(available))
    width, extra = divmod(len(available), count)
    groups, start = [], 0
    for index in range(count):
        stop = start + width + int(index < extra)
        groups.append(tuple(available[start:stop]))
        start = stop
    return groups


def _initialize_worker(group):
    global _BLAS_LIMITS
    os.sched_setaffinity(0, set(group))
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    # NumPy may have loaded before fork; environment variables alone do not
    # resize that live numerical pool. Prediction children inherit both caps.
    _BLAS_LIMITS = threadpool_limits(limits=1, user_api='blas')
    configure_prediction_workers(group)


def _train_job(job):
    try:
        trial = _TRAIN(job)
    finally:
        shutdown_prediction_workers()
    if 'result' not in trial:
        return trial
    return {key: value for key, value in trial.items() if key != 'result'} | {
        'training_ref': str(Path(trial['result']['workspace']) / 'training.json')}


@contextmanager
def checkpoint_execution():
    """Release idle prediction forks and stagger independent large saves.

    Hold this scope while constructing and synchronously writing a checkpoint.
    Scientific work outside the scope keeps its assigned parallel CPU group.
    """
    shutdown_prediction_workers()
    if _CHECKPOINT_LOCK is None:
        yield
    else:
        with _CHECKPOINT_LOCK:
            yield


def run_process_trials(train, jobs, workers, *, release_parent=None):
    """Run independent jobs without queueing their large immutable archives."""
    global _TRAIN, _CHECKPOINT_LOCK
    _TRAIN = train
    context = get_context('fork')
    _CHECKPOINT_LOCK = context.RLock()
    groups = physical_cpu_groups(len(jobs), workers)
    try:
        # Each CPU group owns a worker lifetime. Retire an exhausted group as
        # soon as its last result arrives instead of retaining a completed
        # multi-GiB research scene until the slowest group finishes.
        with ExitStack() as stack:
            executors = [stack.enter_context(ProcessPoolExecutor(1,
                mp_context=context, initializer=_initialize_worker,
                initargs=(group,))) for group in groups]
            pending = dict.fromkeys(range(len(groups)), 0)
            futures = {}
            for index, job in enumerate(jobs):
                group = index % len(groups)
                future = executors[group].submit(_train_job, job)
                futures[future] = (index, group)
                pending[group] += 1
            # The first submit starts each fork worker synchronously. All
            # inherited inputs are now present in their children; the parent
            # no longer needs to keep the admitted scenes resident.
            if release_parent is not None:
                release_parent()
                gc.collect()
            completed = [None] * len(jobs)
            for future in as_completed(futures):
                index, group = futures[future]
                completed[index] = future.result()
                pending[group] -= 1
                if pending[group] == 0:
                    executors[group].shutdown(wait=True)
    finally:
        _TRAIN = None
        _CHECKPOINT_LOCK = None
    return [({**{key: value for key, value in trial.items() if key != 'training_ref'},
              'result': json.loads(Path(trial['training_ref']).read_text())}
             if 'training_ref' in trial else trial)
            for trial in completed]


class ExecutionTiming:
    """Record host diagnostics without changing any scientific fee counter."""

    def __init__(self, phases):
        self.phases = dict.fromkeys(phases, 0.)
        self._wall_start, self._cpu_start = monotonic(), process_time()
        self._utc_start = datetime.now(timezone.utc).isoformat()

    @contextmanager
    def phase(self, name):
        started = monotonic()
        try:
            yield
        finally:
            self.phases[name] += monotonic() - started

    def result(self):
        ended = monotonic()
        return {'wall_seconds': ended - self._wall_start,
            'host_cpu_seconds': process_time() - self._cpu_start,
            'monotonic_start_seconds': self._wall_start,
            'monotonic_end_seconds': ended,
            'utc_start': self._utc_start,
            'utc_end': datetime.now(timezone.utc).isoformat(),
            'utc_clock_scope': 'raw_realtime_anchors_not_the_elapsed_clock',
            'process_id': os.getpid(),
            'wall_clock_scope': 'monotonic_elapsed_including_wait_and_process_suspension',
            'host_cpu_scope': 'current_process_only_excludes_child_processes',
            'phase_seconds': dict(self.phases),
            'after_and_checkpoint_are_nested_in_procedure': True,
            'scope': 'host_wall_diagnostics_not_added_to_scientific_fee_counters'}
