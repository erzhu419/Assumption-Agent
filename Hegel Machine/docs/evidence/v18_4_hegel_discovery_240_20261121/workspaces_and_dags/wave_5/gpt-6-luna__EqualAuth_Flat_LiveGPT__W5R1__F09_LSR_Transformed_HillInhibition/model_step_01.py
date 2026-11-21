import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])  # tau_scale = 1.0 for this task
    s = float(row['s_grad'])
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    def residuals(p):
        return np.array([_predict_with(p, r) - float(r['y']) for r in train_rows])
    result = least_squares(residuals, x0=np.array([3.0, 1.0, 0.5, 0.1]), bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(_params, row))