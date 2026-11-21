import math
import numpy as np
from scipy.optimize import least_squares

_COEF = np.array([0.0, 0.0], dtype=float)

def _basis(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return np.array([math.log1p(u1), u2 / (1.0 + u2)], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_basis(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([1.0, 1.0]))
    _COEF = result.x
    return _COEF

def predict(row) -> float:
    return float(_basis(row) @ _COEF)
