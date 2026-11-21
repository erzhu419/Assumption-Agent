import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, k_half, alpha, c_leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha * s)) - c_leak * u

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float) / 2.5
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    _params = least_squares(lambda p: _model(p, u, s) - y,
                           x0=[max(1.0, 2.0 * np.max(y)), 1.0, 1.0, 0.2],
                           bounds=([0.0, 1e-9, 0.0, 0.0], np.inf),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row):
    return float(_model(_params, float(row['u_now']) / 2.5, float(row['s_grad'])))