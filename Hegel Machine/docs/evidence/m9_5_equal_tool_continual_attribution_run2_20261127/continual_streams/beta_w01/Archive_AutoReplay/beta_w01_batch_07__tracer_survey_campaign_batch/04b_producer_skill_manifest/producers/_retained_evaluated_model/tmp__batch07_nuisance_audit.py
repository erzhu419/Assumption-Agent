import pickle
import runpy
import numpy as np

p = runpy.run_path('/holdout_task/beta_w01_batch_07_solver.py')
with open('public_beta_w01_batch_07.pkl', 'rb') as f:
    cases = pickle.load(f)
worst_profile = worst_equivariance = 0.
errors = []
for case in cases:
    s = case['study']
    banks = np.asarray(s['row_bank_id'])
    S, _ = p['_responses'](s)
    gain, offset = p['_calibration'](s)[banks].T
    y = (s['y_obs_raw'] - offset) / gain
    Z = s['Z_nuisance']
    ridge = s['nuisance_ridge']
    vals = np.column_stack((S, y))
    U, d, _ = np.linalg.svd(Z, full_matrices=False)
    independent = vals - U @ ((d*d/(d*d + ridge))[:, None] * (U.T @ vals))
    actual = p['_profile'](Z, vals, ridge)
    for b in np.unique(banks):
        keep = banks != b
        def fit(profile):
            V = profile(Z[keep], vals[keep], ridge)
            return p['_box_simplex'](V[:, :-1], V[:, -1], s['source_ridge'], s['u_box'], s['total_budget'])[0]
        w = fit(p['_profile'])
        w0 = fit(lambda z, v, r: v)
        R = Z[~keep]
        B = Z[keep]
        beta = np.linalg.solve(B.T @ B + ridge*np.eye(B.shape[1]), B.T @ (y[keep] - S[keep] @ w))
        errors.append((int(b), float(np.sqrt(np.mean((S[~keep] @ w + R @ beta - y[~keep])**2))), float(np.sqrt(np.mean((S[~keep] @ w0 - y[~keep])**2)))))
        worst_profile = max(worst_profile, np.max(np.abs(actual[~keep] - independent[~keep])))
    Q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(Z.shape[1], Z.shape[1])))
    solve = p['solve_beta_w01_batch_07']
    out, rotated = solve(s), solve(dict(s, Z_nuisance=Z @ Q))
    worst_equivariance = max(worst_equivariance, np.max(np.abs(out['weights'] - rotated['weights'])))
print('held-out bank, profiled RMSE, no-nuisance RMSE:', errors)
print(dict(worst_independent_profile_abs=worst_profile, worst_nuisance_rotation_weights_abs=worst_equivariance))
assert worst_profile < 1e-10 and worst_equivariance < 1e-7
