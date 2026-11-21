import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _params
    features = [_features(row) for row in train_rows]
    y = np.array([float(row['y']) for row in train_rows])
    def residual(p):
        return np.array([p[0] * a - p[1] * a * b + p[2] * c for a, b, c in features]) - y
    result = least_squares(residual, [5.0, 1.0, 1.0])
    _params = result.x

def predict(row):
    a, b, c = _features(row)
    return float(_params[0] * a - _params[1] * a * b + _params[2] * c)
