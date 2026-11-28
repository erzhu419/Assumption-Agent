import numpy as np
from scipy.optimize import least_squares


def _saturate(x):
    s = x * x
    return (2.5 * x) * (1.0 + 3.66 * s) / (1.0 + 6.23 * s + 2.01 * s * s)


def _scalar_design(x):
    q = _saturate(x)
    s = x * x
    cross = [q[:, i] * q[:, j] for i in range(4) for j in range(i, 4)]
    return q, np.column_stack((np.ones(len(x)), q, *cross, x + 0.542 * s / (1.0 + 0.164 * s)))


def _scalar_predict(x, y, xt):
    q, base = _scalar_design(x)
    qt, baset = _scalar_design(xt)
    orth = np.linalg.qr(base, mode='reduced')[0]
    target = y - orth @ (orth.T @ y)

    def extra(z, p):
        return np.tanh(z @ p[:4] + p[4])

    def residual(p):
        h = extra(q, p)
        h -= orth @ (orth.T @ h)
        amplitude = (h @ target) / (h @ h + 1e-9)
        return np.r_[target - amplitude * h, 3e-5 * p, 3e-5 * amplitude]

    rng = np.random.default_rng(45)
    starts = [np.full(5, 0.1)] + [rng.normal(0, s, 5) for s in (0.4, 0.8, 1.5) for _ in range(3)]
    best = min((least_squares(residual, p, bounds=(-6, 6), max_nfev=140) for p in starts), key=lambda fit: fit.cost)
    a = np.column_stack((base, extra(q, best.x)))
    at = np.column_stack((baset, extra(qt, best.x)))
    return at @ np.linalg.lstsq(a, y, rcond=None)[0]


def _trajectory_design(x):
    return np.column_stack((np.ones(len(x)), x, np.arctan(2.4 * x)))


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    x, xt = data['z_tr'], data['z_te']
    y_hat = _scalar_predict(x, data['train_y'], xt)
    coef = np.linalg.lstsq(_trajectory_design(x), data['train_traj'], rcond=None)[0]
    state_pred = _trajectory_design(xt) @ coef
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(), state_pred[:, -1].std()])}
