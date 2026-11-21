import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    y = np.array [r['y'] for r in train_rows]
    y = np.asarray(y, dtype=float)
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.05], [100.0, 1.5, 20.0, 100.0, 1000.0])
    best = None
    for tf in (0.1, 0.5, 2.0):
        for ts in (3.0, 15.0, 50.0):
            result = least_squares(lambda p: _evaluate(p, train_rows) - y, [5.0, 0.7, tf, 1.0, ts], bounds=bounds, max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate(_params, [row])[0])