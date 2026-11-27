import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = item['study']
    S, _ = m._forward(s)
    y = m._calibrate(s).ravel()
    Z = s['Z_nuisance']
    nshift, nsensor = len(s['mu_shifts']), len(s['sensor_ij'])
    rows = np.arange(len(y)).reshape(nshift, nsensor)
    w_ref = m.solve_batch(s)['weights']
    ridge_z = s['nuisance_ridge'] * np.eye(Z.shape[1])
    ridge_w = s['source_ridge'] * np.eye(len(w_ref))
    errors = np.empty(rows.shape)
    change = np.empty(nsensor)
    for j in range(nsensor):
        test = rows[:, j]
        train = np.ones(len(y), dtype=bool)
        train[test] = False
        z, a, b = Z[train], S[train], y[train]
        p = np.linalg.solve(z.T @ z + ridge_z, z.T)
        t, r = a - z @ (p @ a), b - z @ (p @ b)
        w, _ = m._box_simplex(t.T @ t + ridge_w, t.T @ r, s['u_box'], s['total_budget'])
        gamma = p @ (b - a @ w)
        errors[:, j] = S[test] @ w + Z[test] @ gamma - y[test]
        change[j] = np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref)
    sensor_rmse = np.sqrt(np.mean(errors**2, axis=0))
    shift_rmse = np.sqrt(np.mean(errors**2, axis=1))
    gamma_full = np.linalg.solve(Z.T @ Z + ridge_z, Z.T @ (y - S @ w_ref))
    full_residual = y - S @ w_ref - Z @ gamma_full
    j = int(sensor_rmse.argmax())
    print('full-fit residual RMSE:', np.sqrt(np.mean(full_residual**2)))
    print('sensor-held-out RMSE by shift:', shift_rmse)
    print('sensor-held-out RMSE median / worst:', np.median(sensor_rmse), sensor_rmse[j], 'at grid index', s['sensor_ij'][j].tolist())
    print('maximum relative weight change:', change.max(), 'at grid index', s['sensor_ij'][change.argmax()].tolist())
    assert np.isfinite(errors).all() and np.isfinite(change).all()
