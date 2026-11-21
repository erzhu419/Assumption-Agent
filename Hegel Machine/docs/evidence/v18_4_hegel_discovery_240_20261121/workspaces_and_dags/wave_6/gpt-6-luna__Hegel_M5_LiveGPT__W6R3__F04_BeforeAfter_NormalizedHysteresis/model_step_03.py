import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    amp, tau = p
    du = float(row['u_now']) - float(row['u_past'])
    h = float(row['H'])
    return float(row['x_now']) + amp * du * math.exp(-h / tau)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([150.0, 1.5]), bounds=([0.0, 1e-6], [1e5, 1e4]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(row, _params))