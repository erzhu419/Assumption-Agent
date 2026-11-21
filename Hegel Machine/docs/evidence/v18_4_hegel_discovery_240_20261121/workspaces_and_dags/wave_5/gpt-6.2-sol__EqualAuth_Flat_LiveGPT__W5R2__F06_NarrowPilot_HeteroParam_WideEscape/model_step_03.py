import math
import numpy as np
from scipy.optimize import least_squares

_coef = None

def _features(row):
    u, s = float(row['u_now']), float(row['s_grad'])
    f = u / (1.2 + u)
    return (f, -f * s / (1.0 + s), math.sqrt(u))

def fit(train_rows):
    global _coef
    rows = list(train_rows)
    X = np.asarray([_features(row) for row in rows], dtype=float)
    y = np.asarray([row['y'] for row in rows], dtype=float)
    _coef = least_squares(lambda p: X @ p - y, np.zeros(3), jac=lambda p: X).x

def predict(row) -> float:
    return float(np.dot(_coef, _features(row)))
