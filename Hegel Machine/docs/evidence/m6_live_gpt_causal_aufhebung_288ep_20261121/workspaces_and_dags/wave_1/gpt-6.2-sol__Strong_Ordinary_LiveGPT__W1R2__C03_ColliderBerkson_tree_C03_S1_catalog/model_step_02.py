import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u2 = float(row['u_2'])
    return math.log1p(float(row['u_1'])), u2 / (1.0 + u2)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.array([_features(row) for row in rows], dtype=float)
    y = np.array([row['y'] for row in rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, np.ones(2), jac=lambda p: X).x
    return _params

def predict(row) -> float:
    return float(np.dot(_params, _features(row)))