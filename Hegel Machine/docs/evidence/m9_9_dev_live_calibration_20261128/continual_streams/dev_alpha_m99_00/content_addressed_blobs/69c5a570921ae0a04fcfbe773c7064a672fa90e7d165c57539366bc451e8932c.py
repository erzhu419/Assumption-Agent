"""Compose the verified bridge calibration with the chamber solver."""
import json
from pathlib import Path

from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream

with Path(__file__).with_name('_upstream_param_payload.json').open() as payload:
    _PAY = json.load(payload)


def solve_batch(dataset: dict) -> dict:
    return solve_downstream(calibrate_upstream(dataset, _PAY))
