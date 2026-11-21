import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)

def _evaluate(p, f):
    a, b, c = f.T
    vmax, rho, local = p
    return vmax * (1.0 - rho * b) * a + local * c

def fit(train_rows):
    global _params
    f = np.array([_features(row) for row in train_rows], dtype=float)
    y = np.array([row['y'] for row in train_rows], dtype=float)
    _params = least_squares(lambda p: _evaluate(p, f) - y, [6.5, 0.70, 0.15], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x
    return _params

def predict(row):
    return float(_evaluate(_params, np.array(_features(row), dtype=float)))