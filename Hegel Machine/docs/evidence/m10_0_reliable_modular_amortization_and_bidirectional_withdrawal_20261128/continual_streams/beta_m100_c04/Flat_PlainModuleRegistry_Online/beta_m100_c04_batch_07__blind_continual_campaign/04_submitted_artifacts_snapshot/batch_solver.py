"""Compose the verified upstream transducer with the chamber solver."""
import json
from pathlib import Path

from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream

_PAY = json.loads(Path(__file__).with_name('_upstream_param_payload.json').read_text())


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study, _PAY))
