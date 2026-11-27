from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(dataset: dict) -> dict:
    # The staged transducer fits this campaign's bridge when no payload is supplied.
    return solve_downstream(calibrate_upstream(dataset))
