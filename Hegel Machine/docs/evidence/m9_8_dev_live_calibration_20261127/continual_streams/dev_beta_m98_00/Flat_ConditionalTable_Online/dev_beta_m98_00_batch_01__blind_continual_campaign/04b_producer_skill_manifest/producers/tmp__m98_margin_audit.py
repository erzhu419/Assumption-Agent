import importlib.util
import pickle
import numpy as np
from itertools import combinations_with_replacement

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)
for item in batch:
    s = item['study']
    c = s['sensor_calibration_sweeps']
    x, y = c['ref_rosette'], c['raw_rosette']
    d = x.shape[1]
    z = (x - x.mean(0)) / x.std(0)
    affine = np.c_[np.ones(len(x)), z]
    quad = np.column_stack([z[:, i] * z[:, j] for i, j in combinations_with_replacement(range(d), 2)])
    def predict(train, test, model):
        if model == 'affine':
            return affine[test] @ np.linalg.lstsq(affine[train], y[train], rcond=None)[0]
        if model == 'pairwise quadratic':
            a = np.c_[affine, quad]
            return a[test] @ np.linalg.lstsq(a[train], y[train], rcond=None)[0]
        # Linear numerator with a denominator quadratic in the four inputs.
        pred = np.empty_like(y[test])
        for j in range(d):
            a = np.c_[affine[train], -y[train, j, None] * quad[train]]
            coef = np.linalg.lstsq(a, y[train, j], rcond=None)[0]
            den = 1 + quad[test] @ coef[d+1:]
            pred[:, j] = affine[test] @ coef[:d+1] / den
        return pred
    print('calibration sweep size:', len(x))
    for model in ['affine', 'pairwise quadratic', 'rational quadratic denominator']:
        errors = []
        for fold in np.array_split(np.random.default_rng(98).permutation(len(x)), 6):
            train = np.setdiff1d(np.arange(len(x)), fold)
            errors.append(predict(train, fold, model) - y[fold])
        rms = [np.sqrt(np.mean(e*e, axis=0)) for e in errors]
        print(model, 'worst held-out fold/channel raw RMSE:', np.max(rms), 'max absolute:', max(np.max(abs(e)) for e in errors))
    S, U = m._design(s)
    ij = s['sensor_ij']
    wall = np.any((ij == 0) | (ij == np.array([s['nx']-1, s['ny']-1])), axis=1)
    emax = {}
    for p in item.get('campaign_diagnostics') or []:
        e = (S @ p['w_pulse'] - m._calibrate({**s, 'y_obs_raw': p['y_pulse_raw']})).reshape(len(s['mu_shifts']), -1)
        for name, mask in [('wall', wall), ('interior', ~wall)]:
            if mask.any():
                emax[name] = max(emax.get(name, 0), np.sqrt(np.mean(e[:, mask]**2, axis=1)).max())
        for bank in range(d):
            key = f'channel {bank}'
            emax[key] = max(emax.get(key, 0), np.sqrt(np.mean(e[:, bank::d]**2, axis=1)).max())
    print('maximum pulse RMSE by shift and slice:', emax)
    print('public studies:', len(batch), 'nuisance columns:', s['Z_nuisance'].shape[1], 'row bank counts:', np.unique(s['row_bank_id'], return_counts=True))
