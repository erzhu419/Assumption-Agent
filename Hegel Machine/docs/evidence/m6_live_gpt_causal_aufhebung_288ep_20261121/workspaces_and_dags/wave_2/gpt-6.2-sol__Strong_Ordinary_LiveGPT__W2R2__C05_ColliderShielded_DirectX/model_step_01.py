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
    A = np.asarray([_features(row) for row in rows], dtype=float)
    y = np.asarray([row['y'] for row in rows], dtype=float)
    _coef = least_squares(lambda b: A @ b - y, np.zeros(2), jac=lambda b: A).x
    return _coef

def predict(row) -> float:
    valve, control = _features(row)
    return math.fsum((_coef[0] * valve, _coef[1] * control))
