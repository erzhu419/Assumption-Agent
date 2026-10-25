"""Parallelize already funded, independent Program predictions in row order.

The research coordinator retains every budget, search cursor and decision. A
configured native worker may give its expensive prediction passes to a persistent
pool on its own CPU group; ordinary callers remain sequential.
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import atexit
import os
from time import perf_counter


_CPUS = ()
_EXECUTOR = None
_ROWS = ()
_CPU_QUEUE = None
_MIN_PARALLEL_SECONDS = .0002
_PREFIX_ROWS = 8


def configure_prediction_workers(cpus):
    """Use only this native task's assigned physical CPU group."""
    global _CPUS
    shutdown_prediction_workers()
    _CPUS = tuple(cpus)


def shutdown_prediction_workers():
    global _EXECUTOR, _ROWS, _CPU_QUEUE
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=True)
        _EXECUTOR = None
    if _CPU_QUEUE is not None:
        _CPU_QUEUE.close()
        _CPU_QUEUE = None
    _ROWS = ()


def _initialize_predictions(rows, cpus):
    global _ROWS, _CPUS, _EXECUTOR, _CPU_QUEUE
    _ROWS = rows
    _CPUS, _EXECUTOR, _CPU_QUEUE = (), None, None
    os.sched_setaffinity(0, {cpus.get()})


def _predict_block(arguments):
    program, start, stop = arguments
    return [program.predict(row) for row in _ROWS[start:stop]]


def _executor(rows):
    global _EXECUTOR, _ROWS, _CPU_QUEUE
    if _EXECUTOR is not None and _ROWS != tuple(rows):
        shutdown_prediction_workers()
    if _EXECUTOR is None:
        _ROWS = tuple(rows)
        context = get_context('fork')
        _CPU_QUEUE = context.SimpleQueue()
        for cpu in _CPUS:
            _CPU_QUEUE.put(cpu)
        _EXECUTOR = ProcessPoolExecutor(max_workers=len(_CPUS), mp_context=context,
            initializer=_initialize_predictions, initargs=(_ROWS, _CPU_QUEUE))
    return _EXECUTOR


def predict_rows(program, rows):
    """Predict each row once; preserve order, floating point sums and failures.

    The first ordinary predictions also measure whether this actual pass is
    expensive enough to benefit. Cheap or short passes stay in the coordinator.
    No budget decision or additional scoring is performed here.
    """
    if len(_CPUS) < 2 or len(rows) <= _PREFIX_ROWS * 2:
        return [program.predict(row) for row in rows]
    started = perf_counter()
    prefix = [program.predict(row) for row in rows[:_PREFIX_ROWS]]
    if perf_counter() - started < _MIN_PARALLEL_SECONDS:
        return [*prefix, *(program.predict(row) for row in rows[_PREFIX_ROWS:])]
    width = (len(rows) - _PREFIX_ROWS + len(_CPUS) - 1) // len(_CPUS)
    blocks = [(program, start, min(start + width, len(rows)))
              for start in range(_PREFIX_ROWS, len(rows), width)]
    for values in _executor(rows).map(_predict_block, blocks, chunksize=1):
        prefix.extend(values)
    return prefix


atexit.register(shutdown_prediction_workers)
