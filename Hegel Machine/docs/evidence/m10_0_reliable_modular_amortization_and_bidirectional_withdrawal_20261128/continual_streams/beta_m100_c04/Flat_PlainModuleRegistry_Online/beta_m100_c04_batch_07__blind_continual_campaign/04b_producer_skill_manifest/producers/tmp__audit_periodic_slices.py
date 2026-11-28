import itertools
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from downstream_solver import _forward
from compare_walls import st, W, Y, Z, axis, predict

cases = list(itertools.product(['open', 'sealed', 'periodic', 'left_open', 'right_open', 'dirichlet', 'mirror', 'face_sealed', 'upwind_sealed'], repeat=2))
periodic = cases.index(('periodic', 'periodic'))
sensors, _ = _forward(st)
pred = np.stack([predict(axis('x', x), axis('y', y)) for x, y in cases])
np.testing.assert_allclose(sensors @ W, pred[periodic], rtol=0, atol=1e-12)
pred[periodic] = sensors @ W
err = pred - Y[None, :, :]
n_sensor = len(st['sensor_ij'])
n_shift = len(st['mu_shifts'])
assert Y.shape[0] == n_sensor * n_shift
rows = np.arange(len(Y))
for grouping, ids in [('shift', rows // n_sensor), ('sensor coordinate', rows % n_sensor), ('rosette channel', rows % 4)]:
    raw_worst, adjusted_worst, smallest_margin = 0., 0., np.inf
    for group in np.unique(ids):
        take = ids == group
        train, test = err[:, ~take, :], err[:, take, :]
        z_train, z_test = Z[~take], Z[take]
        coef = np.linalg.solve(z_train.T @ z_train + st['nuisance_ridge'] * np.eye(Z.shape[1]), z_train.T)
        train_adj = train - z_train @ (coef @ train)
        test_adj = test - z_test @ (coef @ train)
        for tr, te, adjusted in [(train, test, False), (train_adj, test_adj, True)]:
            candidate = np.argmin(np.mean(tr**2, axis=(1, 2)))
            assert candidate == periodic, (grouping, group, adjusted, cases[candidate])
            per_pulse = np.sqrt(np.mean(te**2, axis=1))
            margin = np.delete(per_pulse, periodic, axis=0).min(axis=0) / np.maximum(per_pulse[periodic], 1e-16)
            smallest_margin = min(smallest_margin, margin.min())
            if adjusted:
                adjusted_worst = max(adjusted_worst, per_pulse[periodic].max())
            else:
                raw_worst = max(raw_worst, per_pulse[periodic].max())
    print(f'{grouping}: {len(np.unique(ids))} folds; worst held-out pulse RMSE {raw_worst:.6g} raw, {adjusted_worst:.6g} ridge-adjusted; minimum alternative/periodic RMSE ratio {smallest_margin:.4g}')
    assert max(raw_worst, adjusted_worst) < 1e-5, grouping
