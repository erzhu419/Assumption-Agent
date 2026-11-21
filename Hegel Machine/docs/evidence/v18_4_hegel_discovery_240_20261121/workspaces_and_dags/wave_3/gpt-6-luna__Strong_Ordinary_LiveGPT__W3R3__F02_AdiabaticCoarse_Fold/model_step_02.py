import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p):
    k_ss, a_eff = p
    e = math.exp(-a_eff * float(row['H']))
    u = float(row['u_now'])
    x = float(row['x_now'])
    return k_ss * u + (x - k_ss * u) * e

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_predict_params(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([1.2, 1.0]), bounds=([0.0, 0.0], [10.0, 20.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _params))