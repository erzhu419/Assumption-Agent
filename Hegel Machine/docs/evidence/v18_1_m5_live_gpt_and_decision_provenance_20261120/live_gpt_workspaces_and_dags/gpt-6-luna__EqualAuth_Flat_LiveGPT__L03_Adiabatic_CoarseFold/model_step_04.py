import math
import numpy as np
from scipy.optimize import least_squares

_params = np.array([2.0, 0.5])

def _predict_with(row, params):
    k_ss, a_eff = params
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    e = math.exp(-max(0.0, a_eff) * H)
    return k_ss * u + (x - k_ss * u) * e

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in rows]) - y
    result = least_squares(residual, np.array([2.0, 0.5]), bounds=([0.0, 0.0], [10.0, 10.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_predict_with(row, _params))