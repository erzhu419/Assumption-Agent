import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, rho, c_loc, k_aval, s_crit = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base + c_loc * np.sqrt(u) - k_aval * np.maximum(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    bounds = ([0.0, 0.0, -np.inf, 0.0, 2.44], [np.inf, 1.0, np.inf, np.inf, 2.78])
    best = None
    for crit in (2.5, 2.6, 2.7):
        result = least_squares(lambda p: _model(p, u, s) - y,
                               [8.0, 0.7, 1.0, 2.0, crit],
                               bounds=bounds, max_nfev=10000,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))