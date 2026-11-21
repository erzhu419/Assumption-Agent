import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, u, s):
    vmax, km, rho, c = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * np.sqrt(np.maximum(u, 0.0))

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for km in (0.3, 1.2, 3.0):
        result = least_squares(lambda p: _evaluate(p, u, s) - y, [6.0, km, 0.6, 0.5], bounds=([0.0, 0.001, 0.0, -10.0], [100.0, 100.0, 2.0, 10.0]), max_nfev=5000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, float(row['u_now']), float(row['s_grad'])))