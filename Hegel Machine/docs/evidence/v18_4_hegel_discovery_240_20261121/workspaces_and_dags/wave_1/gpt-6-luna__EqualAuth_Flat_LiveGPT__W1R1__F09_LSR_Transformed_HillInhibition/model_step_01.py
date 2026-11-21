import numpy as np
from scipy.optimize import least_squares

_P = None

def fit(train_rows):
    global _P
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows], dtype=float)
    s = np.array([float(r['s_grad']) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def model(p):
        vmax, k_half, alpha_inh, c_leak = p
        return vmax * (u**2 / (k_half + u**2)) / (1.0 + alpha_inh * s) - c_leak * u
    result = least_squares(lambda p: model(p) - y, x0=np.array([4.0, 1.0, 0.2, 0.05]), bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]))
    _P = tuple(float(v) for v in result.x)
    return _P

def predict(row):
    if _P is None:
        raise ValueError('Call fit(train_rows) before predict(row)')
    vmax, k_half, alpha_inh, c_leak = _P
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (u**2 / (k_half + u**2)) / (1.0 + alpha_inh * s) - c_leak * u)