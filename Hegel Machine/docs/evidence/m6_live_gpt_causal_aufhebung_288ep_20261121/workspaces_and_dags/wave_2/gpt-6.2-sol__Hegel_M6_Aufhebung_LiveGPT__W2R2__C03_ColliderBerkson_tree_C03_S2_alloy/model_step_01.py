import math
import numpy as np
from scipy.optimize import least_squares

_coef = None

def _features(row):
    u2 = float(row['u_2'])
    return math.log1p(float(row['u_1'])), u2 / (1.0 + u2)

def fit(train_rows):
    global _coef
    rows = list(train_rows)
    X = np.array([_features(row) for row in rows], dtype=float)
    y = np.array([float(row['y']) for row in rows], dtype=float)
    _coef = least_squares(lambda a: X @ a - y, np.zeros(2), jac=lambda a: X).x

def predict(row) -> float:
    x1, x2 = _features(row)
    return float(_coef[0] * x1 + _coef[1] * x2)
