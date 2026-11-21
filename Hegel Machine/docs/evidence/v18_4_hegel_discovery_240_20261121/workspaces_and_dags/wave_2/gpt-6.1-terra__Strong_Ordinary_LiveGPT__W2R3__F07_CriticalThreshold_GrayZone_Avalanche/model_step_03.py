import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, rows):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * np.maximum(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    best = None
    for threshold in (2.50, 2.58, 2.68):
        result = least_squares(lambda p: _evaluate(p, rows) - y, [6.0, 0.7, 0.5, 2.0, threshold], bounds=([0.0, 0.0, -20.0, 0.0, 2.50], [100.0, 2.0, 20.0, 100.0, 2.68]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate(_params, [row])[0])
