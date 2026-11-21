import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, params):
    k_ss, a_eff = params
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    z = max(-700.0, min(0.0, -a_eff * H))
    return k_ss * u + (x - k_ss * u) * math.exp(z)

def fit(train_rows):
    global _params
    def residual(params):
        return np.array([_predict_with(r, params) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([2.5, 1.0]), bounds=([0.0, 0.0], [np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(row, _params))