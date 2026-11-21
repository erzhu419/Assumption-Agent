import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 2.5
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_with(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.2, 0.1]), bounds=([1e-8, 1e-8, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(_params, row))