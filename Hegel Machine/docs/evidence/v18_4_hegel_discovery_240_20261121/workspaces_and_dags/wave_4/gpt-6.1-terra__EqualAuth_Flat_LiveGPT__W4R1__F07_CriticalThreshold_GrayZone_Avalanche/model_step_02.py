import numpy as np
from scipy.optimize import least_squares

_params = None

def _eval(p, u, s):
    vmax, vrho, c, k, crit = p
    return (vmax - vrho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(np.maximum(u, 0.0)) - k * np.maximum(0.0, s - crit) * u

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for crit in (2.50, 2.58, 2.68):
        result = least_squares(lambda p: _eval(p, u, s) - y, [5.0, 1.0, 0.5, 2.0, crit], bounds=([0.0, 0.0, -np.inf, 0.0, 2.44], [np.inf, np.inf, np.inf, np.inf, 2.78]), max_nfev=10000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    return float(_eval(_params, float(row['u_now']), float(row['s_grad'])))