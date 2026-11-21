import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([r['u_now'] / 0.1 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def predict_params(p):
        vmax, k_half, alpha_inh, c_leak = p
        return vmax * u**2 / (k_half + u**2) / (1.0 + alpha_inh * s) - c_leak * u
    result = least_squares(lambda p: predict_params(p) - y, x0=[4.0, 1.0, 0.5, 0.05], bounds=([0.0, 1e-9, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    vmax, k_half, alpha_inh, c_leak = _PARAMS
    u = row['u_now'] / 0.1
    s = row['s_grad']
    return float(vmax * u**2 / (k_half + u**2) / (1.0 + alpha_inh * s) - c_leak * u)
