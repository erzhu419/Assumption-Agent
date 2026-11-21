import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return np.array([base, -base * s / (1.0 + s), math.sqrt(u)])

def fit(train_rows):
    global _params
    X = np.asarray([_features(row) for row in train_rows])
    y = np.asarray([float(row['y']) for row in train_rows])
    result = least_squares(lambda p: X @ p - y, np.array([4.0, 2.8, 0.0]))
    _params = result.x
    return _params

def predict(row):
    return float(_features(row) @ _params)
