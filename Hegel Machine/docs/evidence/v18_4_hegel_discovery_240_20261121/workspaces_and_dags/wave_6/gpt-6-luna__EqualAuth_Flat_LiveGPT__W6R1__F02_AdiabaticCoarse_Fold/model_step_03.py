import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p):
    k_ss, a_eff = p
    H = float(row['H'])
    u = float(row['u_now'])
    x = float(row['x_now'])
    e = math.exp(-max(0.0, a_eff) * H)
    return k_ss * u + (x - k_ss * u) * e

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_predict_params(r, p) - float(r['y']) for r in train_rows])
    opt = least_squares(residual, x0=np.array([2.35, 0.5]), bounds=([0.0, 0.0], [10.0, 20.0]))
    _PARAMS = opt.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_params(row, _PARAMS))