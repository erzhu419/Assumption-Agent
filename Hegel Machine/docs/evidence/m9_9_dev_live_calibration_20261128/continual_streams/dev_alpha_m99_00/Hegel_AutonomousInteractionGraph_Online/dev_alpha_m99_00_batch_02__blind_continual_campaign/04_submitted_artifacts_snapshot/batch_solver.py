"""Compose the verified load transducer with the constitutive solver."""
import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_up = _load('_verified_upstream')
_dn = _load('downstream_solver')
_PAY = json.loads((_ROOT / '_upstream_param_payload.json').read_text())


def solve_batch(dataset: dict) -> dict:
    return _dn.solve_downstream(_up.calibrate_upstream(dataset, _PAY))
