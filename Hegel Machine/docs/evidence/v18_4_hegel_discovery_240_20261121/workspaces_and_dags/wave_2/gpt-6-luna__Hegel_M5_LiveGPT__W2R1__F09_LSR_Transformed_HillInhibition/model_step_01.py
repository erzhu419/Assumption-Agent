import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 2.5
_params = None

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.2, 0.1]), bounds=(np.zeros(4), np.full(4, np.inf)), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))