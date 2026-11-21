import math
import numpy as np
from scipy.optimize import least_squares

_coef = None

def _features(row):
    x = float(row['x_ind'])
    return x / (1.1 + x), float(row['u_ctrl'])

def fit(train_rows):
    global _coef
    rows = list(train_rows)
    X = np.array([_features(row) for row in rows], dtype=float)
    y = np.array([row['y'] for row in rows], dtype=float)
    _coef = least_squares(lambda p: X @ p - y, [1.0, 0.1], jac=lambda p: X).x
    return _coef

def predict(row) -> float:
    return math.fsum(c * f for c, f in zip(_coef, _features(row)))
