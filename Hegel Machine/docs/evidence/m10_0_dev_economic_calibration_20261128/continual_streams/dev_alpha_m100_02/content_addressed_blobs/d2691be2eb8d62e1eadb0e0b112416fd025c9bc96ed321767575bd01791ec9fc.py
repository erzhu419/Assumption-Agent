"""Compose per-batch bridge calibration with the chamber solver."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(dataset: dict) -> dict:
    return solve_downstream(calibrate_upstream(dataset, {}))
