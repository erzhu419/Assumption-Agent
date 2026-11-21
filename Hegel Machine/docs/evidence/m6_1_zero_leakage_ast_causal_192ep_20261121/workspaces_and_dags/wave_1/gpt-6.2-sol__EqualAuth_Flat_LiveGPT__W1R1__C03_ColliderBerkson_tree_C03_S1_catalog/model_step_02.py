import math
import numpy as np
from scipy.optimize import least_squares

_coef = None

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return math.log1p(u1), u2 / (1.0 + u2)

def fit(train_rows):
    global _coef
    rows = list(train_rows)
    X = np.array([_features(row) for row in rows], dtype=float)
    y = np.array([row['y'] for row in rows], dtype=float)
    _coef = least_squares(lambda p: X @ p - y, np.ones(2), jac=lambda p: X).x

def predict(row) -> float:
    a, b = _features(row)
    return float(_coef[0] * a + _coef[1] * b)
