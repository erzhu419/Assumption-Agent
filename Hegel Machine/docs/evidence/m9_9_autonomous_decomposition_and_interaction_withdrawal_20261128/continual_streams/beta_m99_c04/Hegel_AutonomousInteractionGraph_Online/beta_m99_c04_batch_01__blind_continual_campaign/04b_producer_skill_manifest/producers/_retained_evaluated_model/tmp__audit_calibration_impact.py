import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
bank = c['single_bank_sweeps']
S, U = m._design(s)
Z = s['Z_nuisance']
Pz = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
P = np.eye(len(S)) - Z @ Pz
A = P @ S
n = A.shape[1]
B = np.eye(n)[:, :-1] - np.eye(n)[:, [-1]]
H = A.T @ A + s['source_ridge'] * np.eye(n)
D = B @ np.linalg.solve(B.T @ H @ B, B.T @ A.T)
w = m.solve_batch(s)['weights']
field_norm = np.linalg.norm(np.tensordot(w, U, axes=1))

for name in ('current', 'rosette_diagonal', 'pooled_diagonal', 'pooled_affine'):
    offset = np.zeros(4)
    mix = np.zeros((4, 4))
    if name != 'current':
        for j in range(4):
            cols = np.arange(4) if name == 'pooled_affine' else [j]
            X = x[:, cols]
            Y = y[:, j]
            if name.startswith('pooled'):
                take = bank['bank_id'] == j
                xs = np.zeros((np.count_nonzero(take), 4))
                xs[:, j] = bank['ref_signal'][take]
                X = np.concatenate((X, xs[:, cols]))
                Y = np.r_[Y, bank['raw_readout'][take]]
            beta = np.linalg.lstsq(np.c_[np.ones(len(X)), X], Y, rcond=None)[0]
            offset[j], mix[cols, j] = beta[0], beta[1:]
    scores = []
    for pulse in item['campaign_diagnostics']:
        raw = pulse['y_pulse_raw']
        cal = (m._calibrate(dict(s, y_obs_raw=raw)) if name == 'current' else
               np.linalg.solve(mix.T, (raw.reshape(-1, 4) - offset).T).T.ravel())
        err = P @ (cal - S @ pulse['w_pulse'])
        dw = D @ err
        df = np.tensordot(dw, U, axes=1)
        scores.append([np.sqrt(np.mean(err.reshape(len(s['mu_shifts']), -1)**2, axis=1)).max(),
                       np.linalg.norm(dw) / np.linalg.norm(w), np.linalg.norm(df) / field_norm])
    print(name, 'pulse rows [worst shift RMS, relative weight perturbation, relative field perturbation]', np.asarray(scores).tolist(), 'worst', np.max(scores, axis=0).tolist())
