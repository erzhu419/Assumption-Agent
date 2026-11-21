import math
import numpy as np
from scipy.optimize import least_squares

_tau_scale = 0.1
_params = None

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / _tau_scale
    s = float(row['s_grad'])
    hill = (u_n * u_n) / (k_half + u_n * u_n)
    return vmax * hill / (1.0 + alpha_inh * s) - c_leak * u_n

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 1.0, 0.2, 0.05]), bounds=([0.0, 1e-9, 0.0, 0.0], [100.0, 1e4, 100.0, 100.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))