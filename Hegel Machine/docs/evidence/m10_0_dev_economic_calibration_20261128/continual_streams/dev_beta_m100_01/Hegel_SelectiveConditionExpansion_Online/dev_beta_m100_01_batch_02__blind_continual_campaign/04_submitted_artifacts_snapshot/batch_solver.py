"""Compose sweep-based sensor calibration with the downstream chamber solver."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study))
