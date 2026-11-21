import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    H = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        E, eta = p
        pred = (u / E) * (1.0 - np.exp(-E * H / eta))
        return pred - y
    result = least_squares(residual, x0=[0.45, 0.8], bounds=([1e-6, 1e-6], [100.0, 1000.0]))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    E, eta = _params
    return float((row['u_now'] / E) * (1.0 - math.exp(-E * row['H'] / eta)))