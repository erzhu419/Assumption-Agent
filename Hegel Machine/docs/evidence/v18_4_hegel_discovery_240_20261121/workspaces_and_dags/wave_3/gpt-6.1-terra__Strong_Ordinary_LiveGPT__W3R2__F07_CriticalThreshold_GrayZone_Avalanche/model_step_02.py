import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, rows):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = np.asarray([r['u_now'] for r in rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in rows], dtype=float)
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * np.maximum(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray [r['y'] for r in rows]
    best = None
    for threshold in (2.50, 2.59, 2.68):
        result = least_squares(lambda p: _values(p, rows) - y, [8.0, 0.7, 0.5, 1.5, threshold], bounds=([0.0, 0.0, -np.inf, 0.0, 2.45], [np.inf, 1.5, np.inf, np.inf, 2.77]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_values(_params, [row])[0])