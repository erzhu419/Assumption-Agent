import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def features(row):
        u = float(row['u_now'])
        s = float(row['s_grad'])
        f = u / (1.2 + u)
        return [f, -f * s / (1.0 + s), math.sqrt(u)]
    X = np.asarray([features(row) for row in train_rows], dtype=float)
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, [7.0, 1.0, -0.3]).x

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    f = u / (1.2 + u)
    return float(_params[0] * f - _params[1] * f * s / (1.0 + s) + _params[2] * math.sqrt(u))