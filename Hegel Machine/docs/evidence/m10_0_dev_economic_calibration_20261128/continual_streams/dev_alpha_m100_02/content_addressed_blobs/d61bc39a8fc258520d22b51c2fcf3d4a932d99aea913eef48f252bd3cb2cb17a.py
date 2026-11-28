"""Calibrate each batch's load bridge, then solve in physical coordinates."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(dataset: dict) -> dict:
    # The verified transducer fits gains and offsets from the supplied sweeps
    # when no precomputed campaign payload is supplied.
    return solve_downstream(calibrate_upstream(dataset, None))
