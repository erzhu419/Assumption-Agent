import math
import numpy as np
from scipy.optimize import least_squares

_fit_params = None

def _response(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    global _fit_params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(logp):
        p = np.exp(logp)
        return np.array([_response(r, p) for r in train_rows]) - y
    result = least_squares(residual, np.log([3.0, 1.0, 0.2, 0.05]), max_nfev=20000)
    _fit_params = np.exp(result.x)
    return _fit_params

def predict(row) -> float:
    if _fit_params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_response(row, _fit_params))