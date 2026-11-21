import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    a, b, c, k, threshold = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return a * base - b * s / (1.0 + s) * base + c * math.sqrt(u) - k * max(0.0, s - threshold) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])
    def residual(p):
        return np.array([_value(row, p) for row in rows]) - y
    candidates = []
    for threshold in (1.31, 1.4, 1.49):
        result = least_squares(residual, [5.0, 1.0, 0.5, 2.0, threshold], bounds=([0.0, 0.0, -np.inf, 0.0, 1.3], [np.inf, np.inf, np.inf, np.inf, 1.5]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        candidates.append(result)
    _params = min(candidates, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return float(_value(row, _params))