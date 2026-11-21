import numpy as np
from scipy.optimize import least_squares


def _response(u, s, p):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha_inh * s)) - c_leak * u


def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([float(r['u_now']) / 0.1 for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    _params = least_squares(
        lambda p: _response(u, s, p) - y,
        x0=[8.0, 2.0, 1.0, 0.3],
        bounds=([0.0, 1e-8, 0.0, 0.0], np.inf),
    ).x


def predict(row):
    return float(_response(float(row['u_now']) / 0.1, float(row['s_grad']), _params))
