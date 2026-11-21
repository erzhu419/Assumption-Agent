import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, rows):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            * (-np.expm1(-h / tau_fast))
            - amp_adapt * u * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [r['y'] for r in rows] if False else np.array([r['y'] for r in rows], dtype=float)
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.1], [1000.0, 2.0, 20.0, 100.0, 1000.0])
    best = None
    for tf in (0.1, 0.5, 1.5):
        for ts in (5.0, 15.0, 40.0):
            result = least_squares(lambda p: _evaluate(p, rows) - y,
                                   [6.0, 0.7, tf, 1.5, ts],
                                   bounds=bounds, max_nfev=5000,
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, [row])[0])