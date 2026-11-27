"""Compose rosette calibration and the downstream chamber solver."""
import json
from pathlib import Path

from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream

_PAY = json.loads(Path(__file__).with_name('_upstream_param_payload.json').read_text())


def solve_batch(study: dict) -> dict:
    # Let the verified transducer fit the current sweeps when supplied.
    payload = None if study.get('sensor_calibration_sweeps') is not None else _PAY
    return solve_downstream(calibrate_upstream(study, payload))
