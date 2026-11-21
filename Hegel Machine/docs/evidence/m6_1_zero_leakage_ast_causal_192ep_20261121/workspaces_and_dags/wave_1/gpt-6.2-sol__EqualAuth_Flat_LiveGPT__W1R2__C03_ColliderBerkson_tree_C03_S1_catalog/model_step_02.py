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
    X = np.asarray([_features(row) for row in rows], dtype=float)
    y = np.asarray([float(row['y']) for row in rows], dtype=float)
    _coef = least_squares(lambda b: X @ b - y, np.zeros(2), jac=lambda b: X).x

def predict(row):
    return float(_coef @ _features(row))
