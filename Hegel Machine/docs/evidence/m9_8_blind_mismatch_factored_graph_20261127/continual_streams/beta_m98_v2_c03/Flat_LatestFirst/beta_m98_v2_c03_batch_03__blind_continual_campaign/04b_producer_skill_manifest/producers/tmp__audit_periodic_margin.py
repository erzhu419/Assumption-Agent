import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
sw = s['sensor_calibration_sweeps']
ref, raw = sw['ref_rosette'], sw['raw_rosette']

def design(x, model):
    terms = [np.ones(len(x)), *x.T]
    if model == 'quadratic':
        terms.extend(x[:, i] * x[:, j] for i in range(4) for j in range(i, 4))
    if model == 'rational':
        terms.extend(x[:, i] / (1 + a * (x[:, i] - x[:, j])**2)
                     for i in range(4) for j in range(4) if i != j for a in (0.25, 1.0))
    return np.column_stack(terms)

# Fit candidate inverse maps on each training fold; score every held-out channel.
folds = np.arange(len(ref)) % 5
scores = {}
for model in ('diagonal', 'affine', 'quadratic', 'rational'):
    parts = []
    for fold in range(5):
        train = folds != fold
        if model == 'diagonal':
            pred = np.column_stack([design(raw[~train, i:i+1], 'affine') @ np.linalg.lstsq(
                design(raw[train, i:i+1], 'affine'), ref[train, i], rcond=None)[0] for i in range(4)])
        else:
            X = design(raw, model)
            pred = X[~train] @ np.linalg.lstsq(X[train], ref[train], rcond=None)[0]
        parts.append(np.sqrt(np.mean((pred - ref[~train])**2, axis=0)))
    scores[model] = float(np.max(parts))
print('worst held-out fold/channel calibration RMSE', scores)
assert scores['affine'] < 3e-5

S, _ = m._forward(s)
ij = np.asarray(s['sensor_ij'])
wall = np.any((ij == 0) | (ij == np.array([s['nx']-1, s['ny']-1])), axis=1)
banks = np.asarray(s['row_bank_id'])
Z = np.asarray(s['Z_nuisance'])
R = Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1])
worst = 0.
for i, pulse in enumerate(item['campaign_diagnostics']):
    delta = m._calibrate(dict(s, y_obs_raw=pulse['y_pulse_raw'])) - S @ pulse['w_pulse']
    gamma = np.linalg.solve(R, Z.T @ delta)
    projected = delta - Z @ gamma
    rmse = lambda x: float(np.sqrt(np.mean(x*x)))
    by_bank = {int(b): rmse(delta[banks == b]) for b in np.unique(banks)}
    by_position = {}
    for j, block in enumerate(delta.reshape(len(s['mu_shifts']), -1)):
        for name, mask in [('wall', wall), ('interior', ~wall)]:
            if np.any(mask):
                by_position[f'{j}/{name}'] = rmse(block[mask])
    worst = max(worst, *by_bank.values(), *by_position.values())
    print('pulse', i, 'unprojected/projected RMSE', rmse(delta), rmse(projected), 'bank RMSE', by_bank, 'shift/wall RMSE', by_position)
assert worst < 5e-6, worst
for a in ('x', 'y'):
    k, v = s['kappa_'+a], s['vel_'+a]
    A = m._axis(k, v, s['L'+a]/(len(k)-1)).toarray()
    assert np.max(np.abs(A.sum(axis=1))) < 1e-12
print('worst absolute pulse slice RMSE', worst)
