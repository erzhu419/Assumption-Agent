import pickle
import runpy
import numpy as np

root = '/holdout_task/'
m = runpy.run_path(root + 'beta_w04_batch_07_solver.py')
cached = np.array([[.7531, -.1118], [1.2928, .1468], [1.7724308310991952, 2.578668096514746], [1.218656964656964, -2.5871185031185027]])
with open(root + 'public_beta_w04_batch_07.pkl', 'rb') as f:
    entries = pickle.load(f)
for e in entries:
    s, gt = e['study'], e['gt']
    D, _ = m['_forward'](s)
    Z, raw, b = s['Z_nuisance'], s['y_obs_raw'], s['row_bank_id']
    p = m['_calibration'](s)
    I = s['nuisance_ridge'] * np.eye(Z.shape[1])
    ridge = s['source_ridge'] * np.eye(D.shape[1])
    X = np.c_[D, (raw-p[b, 1])/p[b, 0]]
    R = X - Z @ np.linalg.solve(Z.T @ Z + I, Z.T @ X)
    G = R.T @ R - np.c_[gt['S_tilde'], gt['r_y']].T @ np.c_[gt['S_tilde'], gt['r_y']]
    h_err, c_err = np.max(abs(G[:-1, :-1])), np.max(abs(G[:-1, -1]))
    print('P^2 source Hessian and linear-term discrepancies:', h_err, c_err)
    assert h_err < 1e-8 and c_err < 1e-7
    ns, nk = len(s['sensor_ij']), len(s['mu_shifts'])
    for order, group in [('shift-major', np.arange(len(b)) % ns), ('sensor-major', np.arange(len(b)) // nk)]:
        scores = []
        for pars in (p, cached):
            y = (raw-pars[b, 1])/pars[b, 0]
            err = np.empty_like(y)
            for j in range(ns):
                test = group == j
                d, z, t = D[~test], Z[~test], y[~test]
                A = np.linalg.solve(z.T @ z + I, z.T @ np.c_[d, t])
                R = np.c_[d, t] - z @ A
                S, r = R[:, :-1], R[:, -1]
                w, _ = m['_box_simplex'](S.T @ S + ridge, -S.T @ r, s['u_box'], s['total_budget'])
                a = A[:, -1] - A[:, :-1] @ w
                prediction = pars[b[test], 0] * (D[test] @ w + Z[test] @ a) + pars[b[test], 1]
                err[test] = (raw[test] - prediction) / p[b[test], 0]
            scores.append(np.array([np.sqrt(np.mean(err[b == j]**2)) for j in range(4)]))
        print(order, 'held-out RMSE by bank (candidate, cache):', *[v.tolist() for v in scores])
        assert np.all(scores[0] < scores[1])
