import math
import numpy as np
from scipy.optimize import least_squares

_coef = None

def _features(row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    f = u / (1.2 + u)
    return (f, -f * s / (1.0 + s), math.sqrt(u))

def fit(train_rows):
    global _coef
    X = np.asarray([_features(row) for row in train_rows], dtype=float)
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    _coef = least_squares(lambda p: X @ p - y, [8.0, 5.6, -0.2], jac=lambda p: X).x

def predict(row) -> float:
    return float(np.dot(_features(row), _coef))
