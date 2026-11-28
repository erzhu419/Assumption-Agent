from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(dataset: dict) -> dict:
    # Omitting a fixed payload invokes the verified per-dataset calibration.
    return solve_downstream(calibrate_upstream(dataset))
