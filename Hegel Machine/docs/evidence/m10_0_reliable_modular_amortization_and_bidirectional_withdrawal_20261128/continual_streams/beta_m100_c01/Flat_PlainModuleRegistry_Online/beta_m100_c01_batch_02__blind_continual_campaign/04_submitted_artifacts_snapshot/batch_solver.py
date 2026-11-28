"""Fit the study's sensor calibration, then solve in calibrated coordinates."""

from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study))
