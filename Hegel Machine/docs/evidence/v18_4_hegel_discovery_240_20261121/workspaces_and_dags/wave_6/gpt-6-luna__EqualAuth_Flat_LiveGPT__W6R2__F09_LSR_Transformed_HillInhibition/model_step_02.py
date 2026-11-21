import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(p, row):
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 1.0, 0.5, 0.05]), bounds=(np.zeros(4), np.full(4, np.inf)))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(_params, row))