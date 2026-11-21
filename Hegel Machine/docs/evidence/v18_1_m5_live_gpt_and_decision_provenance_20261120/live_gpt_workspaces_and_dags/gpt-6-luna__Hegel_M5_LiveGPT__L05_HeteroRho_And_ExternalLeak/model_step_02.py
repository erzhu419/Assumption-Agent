import numpy as np
from scipy.optimize import least_squares

_P = None

def fit(train_rows):
    global _P
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    un = u  # tau_scale = 1.0
    base = un / (1.2 + un)
    def residual(p):
        vmax, rho, branch, leak = p
        pred = vmax * (1.0 - rho * s / (1.0 + s)) * base + branch * x - leak * un * H
        return pred - y
    result = least_squares(residual, x0=[6.5, 0.70, 0.1, 0.1], bounds=([1.0, 0.05, -2.0, -10.0], [15.0, 0.95, 2.0, 10.0]))
    _P = result.x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, rho, branch, leak = _P
    un = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    x = float(row['x_now'])
    H = float(row['H'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + branch * x - leak * un * H)
