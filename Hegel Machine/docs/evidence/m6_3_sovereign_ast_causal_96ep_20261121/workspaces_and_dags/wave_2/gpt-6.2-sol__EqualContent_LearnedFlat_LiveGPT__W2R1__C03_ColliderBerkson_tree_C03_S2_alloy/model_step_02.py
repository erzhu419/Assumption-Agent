import math
import numpy as np
from scipy.optimize import least_squares

_coef = None

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return (math.log1p(u1), u2 / (1.0 + u2))

def fit(train_rows):
    global _coef
    X = np.asarray([_features(row) for row in train_rows], dtype=float)
    y = np.asarray([float(row['y']) for row in train_rows], dtype=float)
    _coef = least_squares(lambda c: X @ c - y, [1.0, 1.0], jac=lambda c: X).x

def predict(row) -> float:
    return float(_coef @ _features(row))