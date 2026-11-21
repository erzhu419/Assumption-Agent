import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def pred(p, row):
        E, tau = p
        return E * (row['u_now'] - row['u_past']) * math.exp(-row['H'] / tau)
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([400.0, 0.5]), bounds=([0.0, 1e-8], [1e6, 1e6]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first')
    E, tau = _params
    return float(E * (row['u_now'] - row['u_past']) * math.exp(-row['H'] / tau))