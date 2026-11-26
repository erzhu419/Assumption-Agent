import importlib.util
import pickle
import numpy as np

p = '/holdout_task/'
def load(name):
    spec = importlib.util.spec_from_file_location(name, p + name + '.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m = load('beta_w03_s00_solver')
# The validator is a CLI and exits when imported. Load only its verifier.
import ast
import math
from pathlib import Path
from types import SimpleNamespace
validator = Path(p + 'validate_public_schema.py')
tree = ast.parse(validator.read_text())
func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'verify_box_simplex_kkt_certificate')
namespace = {'np': np, 'math': math}
exec(compile(ast.Module(body=[func], type_ignores=[]), str(validator), 'exec'), namespace)
v = SimpleNamespace(verify_box_simplex_kkt_certificate=namespace[func.name])
worst = np.zeros(5)
for item in pickle.load(open(p + 'public_beta_w03_s00.pkl', 'rb')):
    s, gt = item['study'], item['gt']
    S, ry = gt['S_tilde'], gt['r_y']
    bank = np.asarray(s['sensor_calibration_sweeps']['bank_id'])
    ref = np.asarray(s['sensor_calibration_sweeps']['ref_signal'])
    raw = np.asarray(s['sensor_calibration_sweeps']['raw_readout'])
    for b in np.unique(bank):
        ix = np.flatnonzero(bank == b)
        errors = np.zeros(5)
        for held in ix:
            train = (bank == b) & (ref != ref[held])
            gain, offset = m._fit_sweep(ref[train], raw[train])
            keep = ~((bank == b) & (ref == ref[held]))
            q = dict(s, sensor_calibration_sweeps={k: np.asarray(a)[keep] for k, a in s['sensor_calibration_sweeps'].items()})
            out = m.solve_beta_w03_s00(q)
            cert = v.verify_box_simplex_kkt_certificate(S, ry, s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
            assert cert['valid'], (b, held, cert)
            a, z = out['weights'], out['terminal_field']
            errors = np.maximum(errors, [abs(gain * ref[held] + offset - raw[held]), np.linalg.norm(a-gt['w_opt'])/np.linalg.norm(gt['w_opt']), np.linalg.norm(z-gt['terminal_field_A'])/np.linalg.norm(gt['terminal_field_A']), cert['gap'], abs(a.sum()-s['total_budget'])])
        print('Bank', b, 'held-out readout error, weights rel-L2, field rel-L2, independent KKT gap, budget error:', errors)
        worst = np.maximum(worst, errors)
print('Worst across real held-out reference levels:', worst)
assert np.all(worst < [1e-6, 1e-5, 1e-5, 1e-6, 1e-7]), worst
