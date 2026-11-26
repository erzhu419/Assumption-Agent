import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
def load(name):
    spec = importlib.util.spec_from_file_location(name, root + name + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

solver = load('beta_w02_batch_04_solver')
validator = load('validate_public_schema')
with open(root + 'public_beta_w02_batch_04.pkl', 'rb') as f:
    case = pickle.load(f)[0]
s, gt = case['study'], case['gt']
sweeps = s['sensor_calibration_sweeps']
b, x, y = map(np.asarray, (sweeps['bank_id'], sweeps['ref_signal'], sweeps['raw_readout']))
base = solver.solve_beta_w02_batch_04(s)
for bank in np.unique(b):
    ids = np.flatnonzero(b == bank)
    held_out = weight_shift = field_shift = gap = 0.0
    for i in ids:
        keep = ids[ids != i]
        g, c = solver._bank_fit(x[keep], y[keep])
        held_out = max(held_out, abs(g*x[i] + c - y[i]))
        for spike in (-0.5, 0.5):
            noisy = y.copy()
            noisy[i] += spike
            out = solver.solve_beta_w02_batch_04(dict(s, sensor_calibration_sweeps=dict(sweeps, raw_readout=noisy)))
            weight_shift = max(weight_shift, np.max(np.abs(out['weights'] - base['weights'])))
            field_shift = max(field_shift, np.linalg.norm(out['terminal_field'] - base['terminal_field']) / np.linalg.norm(base['terminal_field']))
            check = validator.verify_box_simplex_kkt_certificate(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
            assert check['valid'], (bank, i, spike, check)
            gap = max(gap, abs(check['gap']))
    print(f'bank {bank}: held-out error={held_out:.3e}, spike weight delta={weight_shift:.3e}, relative field delta={field_shift:.3e}, independent gap={gap:.3e}', flush=True)
    assert held_out < 1e-7 and max(weight_shift, field_shift) < 1e-6

D, _ = solver._forward(s)
y = solver._calibrate(s)
S, r = solver._profile(s, D, y)
Z = s['Z_nuisance']
F, T = np.column_stack((D, y)), np.column_stack((S, r))
C = Z.T @ F
ref = F.T @ F - C.T @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), C)
err = np.max(np.abs(T.T @ T - ref))
print(f'nuisance Schur-complement max Gram error={err:.3e}', flush=True)
assert err < 1e-10
