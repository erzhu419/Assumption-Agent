import math
import numpy as np
from scipy.optimize import least_squares

_coef = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    sat = u / (1.2 + u)
    return sat, -sat * s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _coef
    rows = list(train_rows)
    design = np.array([_features(row) for row in rows], dtype=float)
    target = np.array([row['y'] for row in rows], dtype=float)
    _coef = least_squares(lambda p: design @ p - target, np.zeros(3), jac=lambda p: design).x

def predict(row) -> float:
    return float(np.dot(_features(row), _coef))
