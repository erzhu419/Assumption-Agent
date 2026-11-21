import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    b = u / (1.2 + u)
    return b, -b * s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _params
    X = np.asarray([_features(row) for row in train_rows], dtype=float)
    y = np.asarray([float(row['y']) for row in train_rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, [7.0, 1.0, 0.0], jac=lambda p: X).x
    return _params

def predict(row) -> float:
    return float(np.dot(_features(row), _params))
