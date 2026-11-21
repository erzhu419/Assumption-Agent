import math
import numpy as np
from scipy.optimize import least_squares

_coef = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    q = u / (1.2 + u)
    return q, -q * s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _coef
    X = np.array([_features(row) for row in train_rows], dtype=float)
    y = np.array([row['y'] for row in train_rows], dtype=float)
    _coef = least_squares(lambda p: X @ p - y, np.ones(3), jac=lambda p: X).x
    return _coef

def predict(row) -> float:
    return float(np.dot(_features(row), _coef))
