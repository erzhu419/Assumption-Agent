import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, rho, strength, threshold = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base - strength * np.maximum(s - threshold, 0.0) ** 2

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for threshold in (1.8, 2.2, 2.5, 2.7):
        result = least_squares(lambda p: _model(p, u, s) - y,
                               [8.0, 0.7, 4.0, threshold],
                               bounds=([0.0, 0.0, 0.0, 0.0],
                                       [100.0, 1.5, 100.0, 3.57]),
                               max_nfev=10000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))