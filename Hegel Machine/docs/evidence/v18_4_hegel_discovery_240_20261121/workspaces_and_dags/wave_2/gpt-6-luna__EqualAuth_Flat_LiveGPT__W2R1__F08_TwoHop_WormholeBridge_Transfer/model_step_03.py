import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    return float(row['u_now']), float(row['s_grad'])

def _model(p, row):
    u, s = _features(row)
    a, b, c, d = p
    return (a + b*u) / (1.0 + c*u) * math.exp(-d*(s - 1.0))

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_model(p, r) for r in train_rows]) - y
    best = None
    for start in ([0.2, 2.0, 0.5, 0.3], [1.0, 2.0, 1.0, 0.1], [0.0, 3.0, 0.8, 0.5]):
        result = least_squares(residual, np.array(start, dtype=float), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    p = fit(train_rows)
    return float(_model(p, row))