from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(dataset: dict) -> dict:
    # Without a cached payload, the verified transducer fits the current sweeps.
    return solve_downstream(calibrate_upstream(dataset))
