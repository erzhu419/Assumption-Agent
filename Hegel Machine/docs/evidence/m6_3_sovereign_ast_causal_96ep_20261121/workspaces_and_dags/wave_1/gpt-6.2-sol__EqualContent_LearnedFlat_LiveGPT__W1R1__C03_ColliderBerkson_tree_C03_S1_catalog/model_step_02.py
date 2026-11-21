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
    X = np.asarray([_features(row) for row in rows], dtype=float)
    y = np.asarray([row['y'] for row in rows], dtype=float)
    _coef = least_squares(lambda c: X @ c - y, [1.0, 1.0], jac=lambda c: X).x

def predict(row):
    x1, x2 = _features(row)
    return float(_coef[0] * x1 + _coef[1] * x2)
