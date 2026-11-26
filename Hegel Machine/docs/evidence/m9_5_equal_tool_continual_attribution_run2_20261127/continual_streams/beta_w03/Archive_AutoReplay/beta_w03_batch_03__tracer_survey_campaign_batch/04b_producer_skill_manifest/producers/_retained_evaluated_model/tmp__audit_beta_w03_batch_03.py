import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m = load('beta_w03_batch_03_solver')
v = load('validate_public_schema')
with (root / 'public_beta_w03_batch_03.pkl').open('rb') as f:
    cases = pickle.load(f)
worst = np.zeros(4)
for q in cases:
    s, g = q['study'], q['gt']
    Z = s['Z_nuisance']; ridge = s['nuisance_ridge']
    i, j = s['sensor_ij'].T
    U = m._responses(s)
    X = U[:, :, i, j].transpose(0, 2, 1).reshape(g['S_tilde'].shape)
    Y = np.column_stack((X, m._calibrate(s)))
    # Invert the full-data ridge residual operator to recover independent
    # unprofiled references; fit the nuisance coefficients anew in each fold.
    R = np.column_stack((g['S_tilde'], g['r_y']))
    Yref = R + Z @ (Z.T @ R) / ridge
    ids = np.tile(np.arange(i.size), U.shape[0])
    for sensor in range(i.size):
        train = ids != sensor
        z = Z[train]
        H = z.T @ z + ridge * np.eye(z.shape[1])
        p = Y[train] - z @ np.linalg.solve(H, z.T @ Y[train])
        r = Yref[train] - z @ np.linalg.solve(H, z.T @ Yref[train])
        worst[0] = max(worst[0], np.max(np.abs(p[:, :-1] - r[:, :-1])))
        worst[1] = max(worst[1], np.max(np.abs(p[:, -1] - r[:, -1])))
        for budget in s['budget_grid']:
            w, cert = m._box_simplex(p[:, :-1], p[:, -1], s['source_ridge'], s['u_box'], float(budget))
            check = v.verify_box_simplex_kkt_certificate(r[:, :-1], r[:, -1], s['source_ridge'], s['u_box'], float(budget), cert)
            assert check['valid'], (sensor, budget, check)
            worst[2] = max(worst[2], check['gap'])
            # Compare held-out predictions after estimating nuisance on training rows only.
            a = np.r_[-w, 1.]
            delta = Y - Yref
            diff = (delta[~train] - Z[~train] @ np.linalg.solve(H, z.T @ delta[train])) @ a
            worst[3] = max(worst[3], np.max(np.abs(diff)))
print('worst leave-one-sensor-out design error, observation error, KKT gap, held-out prediction discrepancy:', worst.tolist(), flush=True)
assert worst[0] < 1e-10 and worst[1] < 1e-7 and worst[3] < 1e-7
