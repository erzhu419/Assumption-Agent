import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(row, k_ss, a_eff):
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    e = math.exp(-a_eff * H)
    return k_ss * u + (x - k_ss * u) * e

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        k_ss, a_eff = p
        return np.asarray([_predict_with(r, k_ss, a_eff) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([1.25, 1.0]), bounds=([0.0, 0.0], [np.inf, np.inf]))
    _PARAMS = (float(result.x[0]), float(result.x[1]))
    return {'k_ss': _PARAMS[0], 'a_eff': _PARAMS[1]}

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(row, *_PARAMS))