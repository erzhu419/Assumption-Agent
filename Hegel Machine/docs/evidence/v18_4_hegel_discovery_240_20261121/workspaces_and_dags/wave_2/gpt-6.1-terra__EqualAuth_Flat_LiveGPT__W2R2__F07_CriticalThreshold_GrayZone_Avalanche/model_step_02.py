import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    a, b, c, k, threshold = p
    base = u / (1.2 + u)
    return a * base - b * base * s / (1.0 + s) + c * np.sqrt(np.maximum(u, 0.0)) - k * np.maximum(0.0, s - threshold) * u

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for threshold in (2.50, 2.59, 2.68):
        result = least_squares(lambda p: _model(p, u, s) - y, [5.0, 1.0, 1.0, 2.0, threshold], bounds=([0.0, 0.0, -np.inf, 0.0, 2.50], [np.inf, np.inf, np.inf, np.inf, 2.68]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))