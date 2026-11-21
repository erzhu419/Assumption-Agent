import math
import numpy as np
from scipy.optimize import least_squares

_params = np.array([0.0, 2.0, 5.0])

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return np.array([1.0, math.log1p(u1), u2 / (1.5 + u2)], dtype=float)

def fit(train_rows):
    global _params
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([0.0, 2.0, 5.0]))
    _params = result.x
    return predict

def predict(row):
    return float(_features(row) @ _params)
