import math
import numpy as np
from scipy.optimize import least_squares

_coef = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return (base, -base * s / (1.0 + s), math.sqrt(u))

def fit(train_rows):
    global _coef
    rows = list(train_rows)
    design = np.array([_features(row) for row in rows])
    y = np.array([float(row['y']) for row in rows])
    result = least_squares(lambda p: design @ p - y, np.zeros(3), jac=lambda p: design)
    _coef = result.x

def predict(row):
    return float(np.dot(_features(row), _coef))