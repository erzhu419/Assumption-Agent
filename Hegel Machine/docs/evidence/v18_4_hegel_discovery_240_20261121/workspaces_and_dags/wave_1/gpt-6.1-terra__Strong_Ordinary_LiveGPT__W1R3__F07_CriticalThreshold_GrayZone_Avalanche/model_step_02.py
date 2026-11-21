import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = p[0] * u / (1.2 + u) - p[1] * s / (1.0 + s) * u / (1.2 + u) + p[2] * math.sqrt(max(0.0, u))
    avalanche = p[3] * max(0.0, s - p[4]) ** p[5]
    return base - avalanche

def fit(train_rows):
    global _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_model(p, r) for r in train_rows]) - y
    best = None
    for threshold in (2.0, 2.3, 2.5, 2.7):
        for exponent in (1.5, 2.0, 3.0):
            result = least_squares(residual, [6.0, 4.2, 0.5, 2.0, threshold, exponent], bounds=([0.0, 0.0, -20.0, 0.0, 0.0, 0.5], [100.0, 100.0, 20.0, 100.0, 3.57, 5.0]), max_nfev=10000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, row))
