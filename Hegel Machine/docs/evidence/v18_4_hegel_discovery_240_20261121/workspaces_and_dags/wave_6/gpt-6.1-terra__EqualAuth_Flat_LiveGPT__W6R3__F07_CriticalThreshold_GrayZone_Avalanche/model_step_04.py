import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    a, b, c, k, threshold = p
    return (a - b * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(max(0.0, u)) - k * max(0.0, s - threshold) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    best = None
    for threshold in (2.50, 2.58, 2.68):
        result = least_squares(residual, [5.0, 1.0, 0.5, 2.0, threshold], bounds=([0.0, 0.0, -np.inf, 0.0, 2.44], [np.inf, np.inf, np.inf, np.inf, 2.78]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))
