import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def predict_params(p, row):
        k_ss, a_eff = p
        u = float(row['u_now'])
        x = float(row['x_now'])
        H = float(row['H'])
        e = math.exp(-a_eff * H)
        return k_ss * u + (x - k_ss * u) * e
    def residuals(p):
        return np.array([predict_params(p, r) - float(r['y']) for r in rows])
    result = least_squares(residuals, x0=np.array([1.0, 0.3]), bounds=([0.0, 0.0], [np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    k_ss, a_eff = _params
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    e = math.exp(-a_eff * H)
    return float(k_ss * u + (x - k_ss * u) * e)
