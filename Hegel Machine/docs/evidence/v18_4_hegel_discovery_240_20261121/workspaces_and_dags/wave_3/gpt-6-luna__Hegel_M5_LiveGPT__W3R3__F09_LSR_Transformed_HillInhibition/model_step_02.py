import math
import numpy as np
from scipy.optimize import least_squares

_fit_params = None

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return vmax * (u_n ** 2 / (k_half + u_n ** 2)) / (1.0 + alpha_inh * s) - c_leak * u_n

def fit(train_rows):
    global _fit_params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([3.0, 1.0, 0.2, 0.02]), bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]))
    _fit_params = result.x
    return _fit_params

def predict(row) -> float:
    if _fit_params is None:
        raise ValueError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _fit_params))
