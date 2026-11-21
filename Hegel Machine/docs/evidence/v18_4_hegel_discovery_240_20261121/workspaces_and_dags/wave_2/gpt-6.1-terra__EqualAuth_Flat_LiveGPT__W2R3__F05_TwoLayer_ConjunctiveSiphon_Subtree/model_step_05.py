import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    a, k, c, d, q = p
    return a * u / (k + u) * (1.0 - d * s / (q + s)) + c * np.sqrt(np.maximum(u, 0.0))

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    bounds = ([0.0, 0.001, 0.0, 0.0, 0.001], [1000.0, 20.0, 100.0, 1.0, 20.0])
    best = None
    for k in (0.1, 1.2):
        for q in (0.3, 1.0, 3.0):
            result = least_squares(lambda p: _model(p, u, s) - y, [20.0, k, 1.0, 0.7, q], bounds=bounds, max_nfev=4000)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))