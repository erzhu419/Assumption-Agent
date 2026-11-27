import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_sibling(name):
    spec = spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_up = _load_sibling('_verified_upstream')
_dn = _load_sibling('downstream_solver')
_PAY = json.loads(Path(__file__).with_name('_upstream_param_payload.json').read_text())


def solve_batch(dataset: dict) -> dict:
    return _dn.solve_downstream(_up.calibrate_upstream(dataset, _PAY))
