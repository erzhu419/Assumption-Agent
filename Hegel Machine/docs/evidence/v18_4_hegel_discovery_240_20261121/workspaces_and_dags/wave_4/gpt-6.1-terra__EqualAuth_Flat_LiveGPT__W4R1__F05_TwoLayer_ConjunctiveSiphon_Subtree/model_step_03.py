import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return np.array([1.0, math.sqrt(u), u, s, s*s, s*math.sqrt(u)])

def fit(train_rows):
    global _params
    X = np.array([_features(row) for row in train_rows])
    y = np.array([float(row['y']) for row in train_rows])
    result = least_squares(lambda p: X @ p - y, np.zeros(X.shape[1]))
    _params = result.x

def predict(row):
    return float(_features(row) @ _params)
