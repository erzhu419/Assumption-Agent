"""Calibrate the study's sensor readings, then solve in canonical coordinates."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    # Omitting the staged payload selects the live sweep fit.
    return solve_downstream(calibrate_upstream(study))
