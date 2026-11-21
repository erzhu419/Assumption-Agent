import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    f = u / (1.2 + u)
    return (f, -f * s / (1.0 + s), math.sqrt(u))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.array([_features(row) for row in rows], dtype=float)
    y = np.array([row['y'] for row in rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, [1.0, 0.5, 0.0], jac=lambda p: X).x

def predict(row):
    return float(np.dot(_features(row), _params))
