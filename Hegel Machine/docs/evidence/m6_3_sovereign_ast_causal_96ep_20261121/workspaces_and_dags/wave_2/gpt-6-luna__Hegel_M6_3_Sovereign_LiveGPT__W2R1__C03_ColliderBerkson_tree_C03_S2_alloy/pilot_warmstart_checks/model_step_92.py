import math
import numpy as np
from scipy.optimize import least_squares
_COEF = np.array([3.0, 4.0], dtype=float)

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return np.array([math.log1p(u1), u2 / (1.0 + u2)], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    res = least_squares(lambda c: X @ c - y, x0=np.array([3.0, 4.0]))
    _COEF = res.x
    return predict

def predict(row) -> float:
    return float(_features(row) @ _COEF)