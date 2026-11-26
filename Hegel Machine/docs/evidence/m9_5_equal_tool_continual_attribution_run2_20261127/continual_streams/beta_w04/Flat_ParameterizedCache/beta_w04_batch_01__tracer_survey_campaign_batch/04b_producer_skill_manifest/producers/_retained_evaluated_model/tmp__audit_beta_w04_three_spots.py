import importlib.util
import itertools
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'beta_w04_batch_01_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open(root + 'public_beta_w04_batch_01.pkl', 'rb') as f:
    data = pickle.load(f)[0]
s, gt = data['study'], data['gt']
sweep = s['sensor_calibration_sweeps']
bank, x, raw = (sweep[k] for k in ('bank_id', 'ref_signal', 'raw_readout'))
rows = s['row_bank_id']
y = solver._calibrate(s)
worst = (0.0, None)
clean_worst = 0.0
for b in (2, 3):
    changed_gain, offset = ((1.27, 0.21) if b == 2 else (0.91, -0.11))
    for sub in itertools.combinations(np.flatnonzero(bank == b), 3):
        keep = (bank != b) | np.isin(np.arange(len(bank)), sub)
        def study_with(readout, obs):
            return dict(s, sensor_calibration_sweeps={'bank_id': bank[keep], 'ref_signal': x[keep], 'raw_readout': readout[keep]}, y_obs_raw=obs)
        shifted = raw.copy()
        shifted[bank == b] = changed_gain * raw[bank == b] + offset
        obs = s['y_obs_raw'].copy()
        obs[rows == b] = changed_gain * obs[rows == b] + offset
        t = study_with(shifted, obs)
        clean_worst = max(clean_worst, np.max(np.abs((solver._calibrate(t) - y)[rows == b])))
        for spike in sub:
            for amplitude in (0.55, -0.8):
                corrupted = raw.copy()
                corrupted[spike] += amplitude
                t = study_with(corrupted, s['y_obs_raw'])
                error = np.max(np.abs((solver._calibrate(t) - y)[rows == b]))
                if error > worst[0]:
                    worst = error, t
print('Worst three-spot EMI corrected-reading change:', worst[0])
print('Worst clean changed-gain three-spot corrected-reading change:', clean_worst)
assert worst[0] < 1e-7 and clean_worst < 1e-5
out = solver.solve_beta_w04_batch_01(worst[1])
weights = out['weights_by_budget']
werr = np.max(np.linalg.norm(weights - gt['weights_by_budget'], axis=1) / np.linalg.norm(gt['weights_by_budget'], axis=1))
field = out['terminal_field_mid_budget']
ferr = np.linalg.norm(field - gt['terminal_field_mid_budget']) / np.linalg.norm(gt['terminal_field_mid_budget'])
S, r = gt['S_tilde'], gt['r_y']
gaps = []
for w, total, cert in zip(weights, s['budget_grid'], out['kkt_certificates']):
    cap, nu = s['u_box'], cert['nu_sum']
    g = S.T @ (S @ w - r) + s['source_ridge'] * w
    gaps.append(abs(g @ w - nu * total - cap * np.minimum(g - nu, 0.0).sum()))
    assert abs(w.sum() - total) < 1e-7 and w.min() >= -1e-7 and w.max() <= cap + 1e-7
print('Worst EMI inversion relative errors (weights, terminal field) and public-design KKT gaps:', werr, ferr, gaps)
assert max(werr, ferr, *gaps) < 1e-6
