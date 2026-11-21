import math
import numpy as np
from scipy.optimize import least_squares

_FITTED_PARAMS = None

def _predict_params(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (u * u / (k_half + u * u)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _FITTED_PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.2, 0.1]), bounds=(np.zeros(4), np.full(4, np.inf)), max_nfev=20000)
    _FITTED_PARAMS = result.x
    return result.x

def predict(row):
    if _FITTED_PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _FITTED_PARAMS))