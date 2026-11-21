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
    a = np.asarray([_features(row) for row in rows], dtype=float)
    y = np.asarray([float(row['y']) for row in rows], dtype=float)
    _coef = least_squares(lambda p: a @ p - y, np.zeros(2), jac=lambda p: a).x

def predict(row) -> float:
    x_term, u = _features(row)
    return math.fsum((_coef[0] * x_term, _coef[1] * u))
