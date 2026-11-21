import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (u * u / (k_half + u * u)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    x0 = np.array([3.0, 1.0, 0.5, 0.1], dtype=float)
    result = least_squares(
        lambda p: np.asarray([_predict_with(p, r) for r in train_rows]) - y,
        x0=x0,
        bounds=([1e-9, 1e-9, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]),
        max_nfev=20000,
    )
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(_params, row))