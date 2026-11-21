import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u2 = u * u
    return vmax * u2 / (k_half + u2) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 1.0, 0.2, 0.05]), bounds=([0.0, 1e-8, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))