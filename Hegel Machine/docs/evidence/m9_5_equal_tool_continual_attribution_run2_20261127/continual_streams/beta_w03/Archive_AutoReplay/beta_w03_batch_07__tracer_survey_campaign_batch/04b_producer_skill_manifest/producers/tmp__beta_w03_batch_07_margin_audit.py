import importlib.util
import json
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_w03_batch_07_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_beta_w03_batch_07.pkl', 'rb') as f:
    cases = pickle.load(f)
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    history = json.load(f)

for case in cases:
    s, gt = case['study'], case['gt']
    u = m._responses(s)
    i, j = np.asarray(s['sensor_ij']).T
    raw = np.asarray(s['y_obs_raw'])
    banks = np.asarray(s['row_bank_id'])
    d = u[:, :, i, j].transpose(0, 2, 1).reshape(len(raw), u.shape[1])
    z = np.asarray(s['Z_nuisance'])
    lam = s['nuisance_ridge']
    ridge, cap, budget = (s[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    y = m._calibrate(s)
    deployed = m.solve_beta_w03_batch_07(s)
    a, r = gt['S_tilde'], gt['r_y']
    def tangent_gap(w):
        g = a.T @ (a @ w - r) + ridge * w
        order = np.argsort(g)
        v = np.zeros_like(w)
        v[order] = np.clip(budget - cap * np.arange(len(w)), 0, cap)
        return float(g @ (w-v))
    for record in history:
        hb, x, q = (np.asarray(record[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
        if np.max(np.abs(q-x), initial=0.) < 1e-3 or any(np.sum(hb == b) < 8 for b in range(4)):
            continue
        yy = y.copy()
        for b in (2, 3):
            gain, offset = np.polyfit(x[hb == b], q[hb == b], 1)
            yy[banks == b] = (raw[banks == b] - offset) / gain
        p = m._profile(np.column_stack((d, yy)), z, lam)
        w, _ = m._box_simplex(p[:, :-1], p[:, -1], ridge, cap, budget)
        field = np.einsum('m,mxy->xy', w, u[-1])
        wr = np.linalg.norm(w-gt['w_opt']) / np.linalg.norm(gt['w_opt'])
        fr = np.linalg.norm(field-gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A'])
        gap = tangent_gap(w)
        print(record['episode_short_name'], 'weight relative L2', wr, 'field relative L2', fr, 'public-objective tangent gap', gap)
        assert max(wr, fr) < 1e-5 and gap < 1e-6
    # Predict the held physical sensors at every shift, without using their observations for either fit.
    sensor = np.tile(np.arange(len(i)), len(s['mu_shifts']))
    scores = np.empty((len(i), 2))
    for k in range(len(i)):
        test = sensor == k
        train = ~test
        ztr = z[train]
        aug = np.vstack((ztr, np.sqrt(lam)*np.eye(z.shape[1])))
        for mode in range(2):
            response = y if mode == 0 else np.where(banks >= 2, raw, y)
            p = m._profile(np.column_stack((d[train], response[train])), ztr, lam)
            w, _ = m._box_simplex(p[:, :-1], p[:, -1], ridge, cap, budget)
            coeff = np.linalg.lstsq(aug, np.r_[response[train]-d[train]@w, np.zeros(z.shape[1])], rcond=None)[0]
            resid = y[test] - d[test] @ w - z[test] @ coeff
            scores[k, mode] = np.sqrt(np.mean(resid**2))
    print('sensor jackknife worst RMSE, archive versus identity training:', scores.max(axis=0))
    print('sensor jackknife median RMSE, archive versus identity training:', np.median(scores, axis=0))
    print('sensor jackknife folds improved by archive:', int(np.sum(scores[:,0] < scores[:,1])), '/', len(scores))
    print('deployed public-objective tangent gap:', tangent_gap(deployed['weights']))
