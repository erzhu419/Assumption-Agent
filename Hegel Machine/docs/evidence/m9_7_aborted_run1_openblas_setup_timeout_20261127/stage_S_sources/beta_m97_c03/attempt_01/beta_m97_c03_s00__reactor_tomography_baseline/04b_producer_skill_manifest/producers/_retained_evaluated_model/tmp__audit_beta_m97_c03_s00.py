import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'beta_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_beta_m97_c03_s00.pkl').open('rb') as f:
    cases = pickle.load(f)
worst_holdout = worst_shift = worst_control = worst_robin = 0.
for case in cases:
    s, gt = case['study'], case['gt']
    sw = s['sensor_calibration_sweeps']
    S, _ = m._forward(s)
    Z = s['Z_nuisance']
    gram = Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1])
    def project(a):
        return a - Z @ np.linalg.solve(gram, Z.T @ a)
    y = m._calibrate(s)
    ps = project(S)
    args = tuple(s[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    baseline = m.solve_beta_m97_c03_s00(s)['weights']
    control, _ = m._box_simplex(S, y, *args)
    worst_control = max(worst_control, np.max(np.abs(control - gt['w_opt'])))
    for bank in range(4):
        ids = np.flatnonzero(sw['bank_id'] == bank)
        x, raw = sw['ref_signal'][ids], sw['raw_readout'][ids]
        for i in range(ids.size):
            keep = np.arange(ids.size) != i
            gain, offset = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), raw[keep], rcond=None)[0]
            worst_holdout = max(worst_holdout, abs(raw[i] - (gain*x[i] + offset)))
            y_jk = y.copy()
            mask = s['row_bank_id'] == bank
            y_jk[mask] = (s['y_obs_raw'][mask] - offset) / gain
            w, _ = m._box_simplex(ps, project(y_jk), *args)
            worst_shift = max(worst_shift, np.max(np.abs(w - baseline)))
    for axis in ('x', 'y'):
        n = s['n' + axis]
        h = s['L' + axis] / (n - 1)
        rob = s['robin_' + axis]
        A = m._axis_operator(s['kappa_' + axis], s['vel_' + axis], rob, h)
        expected = np.zeros(n)
        expected[[0, -1]] = rob / h
        worst_robin = max(worst_robin, np.max(np.abs(A @ np.ones(n) - expected)))
print('Worst held-out calibration residual:', worst_holdout)
print('Worst weight change from omitting one measured sweep row:', worst_shift)
print('Worst weight error with nuisance projection omitted (negative control):', worst_control)
print('Worst constant-field axis stencil error:', worst_robin)
assert worst_holdout < 1e-6 and worst_shift < 1e-6 and worst_robin < 1e-12
