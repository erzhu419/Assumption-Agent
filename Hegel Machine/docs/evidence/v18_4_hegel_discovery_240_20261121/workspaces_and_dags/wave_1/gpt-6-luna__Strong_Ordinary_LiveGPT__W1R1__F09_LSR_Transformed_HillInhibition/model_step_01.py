import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (u * u / (k_half + u * u)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(logp):
        p = np.exp(logp)
        return np.array([_value(r, p) for r in rows]) - y
    result = least_squares(residual, np.log([4.0, 1.0, 0.2, 0.05]), max_nfev=20000)
    _params = np.exp(result.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _params))