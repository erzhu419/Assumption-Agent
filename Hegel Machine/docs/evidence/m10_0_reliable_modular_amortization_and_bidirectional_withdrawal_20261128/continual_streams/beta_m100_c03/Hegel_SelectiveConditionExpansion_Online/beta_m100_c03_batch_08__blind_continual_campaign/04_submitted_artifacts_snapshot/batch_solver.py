"""Calibrate the raw study, then solve the chamber inverse problem."""
import inspect
import json
from pathlib import Path

from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream

_NEEDS_PAYLOAD = len(inspect.signature(calibrate_upstream).parameters) >= 2
if _NEEDS_PAYLOAD:
    _PAYLOAD = json.loads(Path(__file__).with_name('_upstream_param_payload.json').read_text())


def solve_batch(study: dict) -> dict:
    calibrated = calibrate_upstream(study, _PAYLOAD) if _NEEDS_PAYLOAD else calibrate_upstream(study)
    return solve_downstream(calibrated)
