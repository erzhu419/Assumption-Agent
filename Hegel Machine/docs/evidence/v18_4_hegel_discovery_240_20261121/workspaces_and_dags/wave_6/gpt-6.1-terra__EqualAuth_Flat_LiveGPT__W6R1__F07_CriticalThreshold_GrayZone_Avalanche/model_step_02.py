import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    a, b, c, k, threshold = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return (a - b * s / (1.0 + s)) * base + c * math.sqrt(u) - k * max(0.0, s - threshold) * u

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(row, p) - float(row['y']) for row in train_rows])
    result = least_squares(residual, [5.0, 1.0, 0.5, 2.0, 2.6], bounds=([0.0, 0.0, 0.0, 0.0, 2.44], [np.inf, np.inf, np.inf, np.inf, 2.78]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    return float(_value(row, _params))