import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    sink = 1.0 - rho * s / (1.0 + s)
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * sink * u / (1.2 + u) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    lower = [0.0, 0.0, 0.001, 0.0, 0.05]
    upper = [100.0, 1.5, 20.0, 20.0, 500.0]
    best = None
    for tf in (0.1, 0.4, 1.0):
        for ts in (3.0, 10.0, 30.0):
            initial = [6.0, 0.7, tf, 1.0, ts]
            result = least_squares(lambda p: _evaluate(p, u, s, h) - y,
                                   initial, bounds=(lower, upper),
                                   max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, float(row['u_now']), float(row['s_grad']), float(row['H'])))
