import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    drive = u / (1.2 + u)
    return (drive, -drive * s / (1.0 + s), math.sqrt(u))


def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.array([_features(row) for row in rows], dtype=float)
    y = np.array([row['y'] for row in rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, np.ones(3), jac=lambda p: X).x
    return _params


def predict(row):
    return float(np.dot(_features(row), _params))
