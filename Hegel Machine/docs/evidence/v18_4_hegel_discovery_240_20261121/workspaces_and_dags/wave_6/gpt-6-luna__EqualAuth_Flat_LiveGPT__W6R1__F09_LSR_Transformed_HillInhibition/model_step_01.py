import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    return u, s

def _predict_params(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u, s = _features(row)
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residuals(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residuals, x0=np.array([3.0, 1.0, 0.2, 0.05]), bounds=(np.zeros(4), np.full(4, np.inf)), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _params))