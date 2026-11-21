import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)

def _evaluate(p, a, b, c):
    vmax, rho, loc = p
    return vmax * (1.0 - rho * b) * a + loc * c

def fit(train_rows):
    global _params
    a, b, c = np.asarray([_features(r) for r in train_rows], dtype=float).T
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    _params = least_squares(lambda p: _evaluate(p, a, b, c) - y, x0=[6.5, 0.70, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x

def predict(row) -> float:
    return float(_evaluate(_params, *_features(row)))