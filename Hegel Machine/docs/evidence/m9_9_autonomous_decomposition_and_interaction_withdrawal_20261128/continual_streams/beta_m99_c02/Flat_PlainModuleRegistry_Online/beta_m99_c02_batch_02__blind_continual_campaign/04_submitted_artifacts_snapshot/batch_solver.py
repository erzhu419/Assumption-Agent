"""Calibrate the current study and solve it in canonical coordinates."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    # With no fixed payload, the verified transducer fits the current sweeps.
    return solve_downstream(calibrate_upstream(study))
