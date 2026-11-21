import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    a, log_b, c, d, log_e = p
    b = math.exp(log_b)
    e = math.exp(log_e)
    return a * math.log1p(b * row['u_now']) + c - d * math.log(row['s_grad'] + e)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(r, p) - r['y'] for r in train_rows])
    best = None
    for b in (0.5, 2.0, 6.0):
        for e in (0.1, 0.3, 1.0):
            result = least_squares(residual, [1.5, math.log(b), 1.0, 1.0, math.log(e)], bounds=([-20, -8, -20, -20, -8], [20, 8, 20, 20, 8]), max_nfev=3000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))