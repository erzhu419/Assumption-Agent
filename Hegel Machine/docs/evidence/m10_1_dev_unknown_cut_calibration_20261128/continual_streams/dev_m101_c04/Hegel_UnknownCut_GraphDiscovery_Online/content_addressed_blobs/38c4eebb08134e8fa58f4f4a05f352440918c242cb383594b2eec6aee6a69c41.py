from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


def _quadratic(x):
    return np.column_stack((np.ones(len(x)), x, *(
        x[:, i] * x[:, j]
        for i, j in combinations_with_replacement(range(x.shape[1]), 2)
    )))


def _scalar(q, z, y, q_test, z_test):
    # Profile out the quadratic and saturation amplitude while fitting the
    # direction and offset of a single saturation in the mixed coordinates.
    center, scale = z.mean(axis=0), z.std(axis=0)
    a = np.column_stack((np.ones(len(z)), (z - center) / scale))
    a_test = np.column_stack((np.ones(len(z_test)), (z_test - center) / scale))
    basis, _ = np.linalg.qr(q, mode='reduced')
    residual = y - basis @ (basis.T @ y)

    def objective(p):
        h = np.tanh(a @ p)
        h -= basis @ (basis.T @ h)
        return h * ((h @ residual) / max(h @ h, 1e-20)) - residual

    rng = np.random.default_rng(442)
    starts = rng.normal(0.0, 0.7, (16, a.shape[1]))
    optimum = min((least_squares(objective, start, bounds=(-5.0, 5.0),
                                 max_nfev=160) for start in starts),
                  key=lambda fit: fit.cost).x
    features = np.column_stack((q, np.tanh(a @ optimum)))
    beta = np.linalg.lstsq(features, y, rcond=None)[0]
    return q_test @ beta[:-1] + beta[-1] * np.tanh(a_test @ optimum)


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    up, mixed = inputs['node_02'], inputs['node_00']
    q, q_test = _quadratic(up['u_tr']), _quadratic(up['u_te'])
    state = q_test @ np.linalg.lstsq(q, train['traj'], rcond=None)[0]
    y = _scalar(q, mixed['z_tr'], np.asarray(train['y']), q_test, mixed['z_te'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
