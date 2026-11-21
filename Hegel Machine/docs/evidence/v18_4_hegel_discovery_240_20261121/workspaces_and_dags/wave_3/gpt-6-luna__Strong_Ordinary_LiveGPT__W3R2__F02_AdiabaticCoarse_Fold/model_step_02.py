import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, params):
    k_ss, a_eff = params
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    return k_ss * u + (x - k_ss * u) * math.exp(-a_eff * H)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    if not rows:
        _params = (1.75, 1.0)
        return _params
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.75, 1.0]), bounds=([0.0, 0.0], [100.0, 100.0]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))