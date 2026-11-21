import numpy as np
from scipy.optimize import least_squares
import math

_PARAMS = None

def _predict_row(row, p):
    k_ss, a_eff = p
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    x_ss = k_ss * u
    return x_ss + (x - x_ss) * math.exp(-a_eff * H)

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_predict_row(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([2.0, 0.05]), bounds=([0.0, 0.0], [10.0, 10.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_row(row, _PARAMS))