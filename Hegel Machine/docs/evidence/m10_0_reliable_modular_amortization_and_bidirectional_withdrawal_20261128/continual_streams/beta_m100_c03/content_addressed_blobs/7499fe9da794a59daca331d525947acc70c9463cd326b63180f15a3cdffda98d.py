from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    """Calibrate from the study's sweeps, then solve in canonical coordinates."""
    return solve_downstream(calibrate_upstream(study))
