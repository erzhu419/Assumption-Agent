import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    v = max(float(row['u_now']) / 0.005, 0.0)
    s = max(float(row['s_grad']), 0.0)
    a, k, b, c, q, power = p
    substrate = v / (k + v)
    sp = s ** power
    gradient = sp / (q ** power + sp)
    return a * substrate + b * math.sqrt(v) - c * substrate * gradient

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    lower = [0.0, 0.02, 0.0, 0.0, 0.02, 0.3]
    upper = [30.0, 30.0, 10.0, 30.0, 20.0, 4.0]
    starts = [[5, 1.2, 0.5, 2, 1, 1], [8, 3, 0.2, 3, 0.5, 2], [3, 0.5, 1, 1, 2, 1]]
    fits = [least_squares(residual, p, bounds=(lower, upper), max_nfev=10000) for p in starts]
    _params = min(fits, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return float(_value(row, _params))
