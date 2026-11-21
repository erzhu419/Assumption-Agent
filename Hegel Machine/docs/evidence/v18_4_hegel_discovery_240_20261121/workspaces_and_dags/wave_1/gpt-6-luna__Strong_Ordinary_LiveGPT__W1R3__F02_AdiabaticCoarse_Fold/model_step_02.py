import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def pred(p, r):
        k_ss, a_eff = p
        e = math.exp(-a_eff * float(r['H']))
        return k_ss * float(r['u_now']) + (float(r['x_now']) - k_ss * float(r['u_now'])) * e
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([1.86, 0.01]), bounds=([0.0, 0.0], [10.0, 10.0]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    k_ss, a_eff = _PARAMS
    u = float(row['u_now'])
    e = math.exp(-a_eff * float(row['H']))
    return float(k_ss * u + (float(row['x_now']) - k_ss * u) * e)
