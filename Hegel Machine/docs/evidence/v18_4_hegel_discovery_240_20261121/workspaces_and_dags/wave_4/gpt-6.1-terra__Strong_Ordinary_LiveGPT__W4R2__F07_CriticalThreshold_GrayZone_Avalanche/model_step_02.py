import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, u, s):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * np.maximum(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    lower = [0.0, 0.0, -20.0, 0.0, 2.44]
    upper = [100.0, 2.0, 20.0, 100.0, 2.78]
    best = None
    for threshold in (2.5, 2.6, 2.7):
        for rho in (0.5, 0.7, 1.0):
            result = least_squares(lambda p: _evaluate(p, u, s) - y, [8.0, rho, 1.0, 1.5, threshold], bounds=(lower, upper), max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, float(row['u_now']), float(row['s_grad'])))