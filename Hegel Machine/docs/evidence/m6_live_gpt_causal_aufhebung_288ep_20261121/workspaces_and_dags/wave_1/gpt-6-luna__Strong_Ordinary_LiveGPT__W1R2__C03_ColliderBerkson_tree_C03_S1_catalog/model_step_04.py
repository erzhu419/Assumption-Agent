import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    a, b, k = p
    return a * math.log1p(u1) + b * u2 / (k + u2)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    if not rows:
        raise ValueError('train_rows must not be empty')
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([2.0, 3.0, 1.5]), bounds=([-20.0, -20.0, 1e-6], [20.0, 20.0, 100.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))