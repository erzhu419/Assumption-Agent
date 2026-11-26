import json
import pickle
import runpy
from pathlib import Path

import numpy as np

root = Path('/holdout_task')
m = runpy.run_path(str(root / 'beta_w04_batch_07_solver.py'))
archive = Path('/tmp/hegel_continual_archive')
history = json.loads((archive / 'archive_calibration_history.json').read_text())
cached = np.array([[.7531, -.1118], [1.2928, .1468], [1.7724308310991952, 2.578668096514746], [1.218656964656964, -2.5871185031185027]])
for e in pickle.loads((root / 'public_beta_w04_batch_07.pkl').read_bytes()):
    s, gt = e['study'], e['gt']
    v = s['V_reaction_2d']
    mixed = np.diff(np.diff(v, axis=0), axis=1)
    print('max mixed reaction difference:', np.max(abs(mixed)))
    assert np.linalg.norm(mixed) <= 1e-6 * np.linalg.norm(v)
    params = m['_calibration'](s)
    q = s['sensor_calibration_sweeps']
    qb, x, y = (np.asarray(q[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    for episode in ('beta_w04_batch_02', 'beta_w04_batch_04'):
        h = next(h for h in history if h['episode_short_name'] == episode)
        hb, hx, hy = (np.asarray(h[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
        scores = []
        for b in (2, 3):
            k = hb == b
            p = np.linalg.lstsq(np.c_[hx[k], np.ones(k.sum())], hy[k], rcond=None)[0]
            k = qb == b
            r = y[k] - (p[0]*x[k] + p[1])
            scores.append(float(np.max(abs(r - np.round(r, 1)))))
            assert np.min(abs(r)) < 3e-6 and scores[-1] < 3e-6
        print('independent archive episode', episode, 'Wing B corrected residuals:', scores)
    design, _ = m['_forward'](s)
    z, b = s['Z_nuisance'], s['row_bank_id']
    raw = s['y_obs_raw']
    rho, lam = s['nuisance_ridge'], s['source_ridge']
    sensor_count = len(s['sensor_ij'])
    for order, group in [('shift-major', np.arange(len(b)) % sensor_count), ('sensor-major', np.arange(len(b)) // len(s['mu_shifts']))]:
        bank_rmse = []
        for p in (params, cached):
            y = (raw-p[b, 1])/p[b, 0]
            error = np.empty_like(y)
            for j in range(sensor_count):
                test = group == j
                train = ~test
                d, zz, yy = design[train], z[train], y[train]
                N = zz.T @ zz + rho*np.eye(z.shape[1])
                pd = d - zz @ np.linalg.solve(N, zz.T @ d)
                py = yy - zz @ np.linalg.solve(N, zz.T @ yy)
                w, _ = m['_box_simplex'](d.T @ pd + lam*np.eye(design.shape[1]), -d.T @ py, s['u_box'], s['total_budget'])
                a = np.linalg.solve(N, zz.T @ (yy-d@w))
                error[test] = (raw[test] - (p[b[test], 0]*(design[test]@w + z[test]@a) + p[b[test], 1])) / params[b[test], 0]
            bank_rmse.append([float(np.sqrt(np.mean(error[b==j]**2))) for j in range(4)])
        print(order, 'held-out RMSE by bank (current, cached), common scale:', bank_rmse)
    y = (raw-params[b, 1])/params[b, 0]
    N = z.T @ z + rho*np.eye(z.shape[1])
    P = np.eye(len(b))-z @ np.linalg.solve(N, z.T)
    assert np.max(abs(design.T@P@design - gt['S_tilde'].T@gt['S_tilde'])) < 1e-8
    assert np.max(abs(design.T@P@y - gt['S_tilde'].T@gt['r_y'])) < 1e-7
    print('Independent ridge Schur complement matches the public quadratic and linear terms.')
