import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    sat = u / (1.2 + u)
    return (sat, -sat * s / (1.0 + s), math.sqrt(u))

def fit(train_rows):
    global _params
    X = np.asarray([_features(row) for row in train_rows], dtype=float)
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, np.ones(3), jac=lambda p: X).x
    return _params

def predict(row) -> float:
    return float(np.dot(_features(row), _params))