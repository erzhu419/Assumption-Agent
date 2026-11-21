import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, rows):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [r['y'] for r in rows] if False else np.array([r['y'] for r in rows], dtype=float)
    best = None
    for tf in (0.2, 0.5, 1.0):
        for ts in (5.0, 15.0, 40.0):
            result = least_squares(lambda p: _evaluate(p, rows) - y, [6.5, 0.7, tf, 0.8, ts], bounds=([0.0, 0.0, 0.001, 0.0, 0.01], [100.0, 1.0, 100.0, 100.0, 1000.0]), max_nfev=5000)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate(_params, [row])[0])