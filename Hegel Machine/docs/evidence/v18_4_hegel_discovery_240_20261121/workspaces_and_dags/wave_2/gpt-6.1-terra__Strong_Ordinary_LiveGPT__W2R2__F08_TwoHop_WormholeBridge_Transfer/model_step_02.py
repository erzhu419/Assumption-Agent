import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u) + p[2] * math.sqrt(u)

def fit(train_rows):
    global _params
    result = least_squares(lambda p: np.array([_value(r, p) - float(r['y']) for r in train_rows]), [6.0, 0.7, 0.5], max_nfev=10000)
    _params = result.x

def predict(row):
    return float(_value(row, _params))