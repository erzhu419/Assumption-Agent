import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, rows):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.asarray([r['u_now'] for r in rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in rows], dtype=float)
    h = np.asarray([r['H'] for r in rows], dtype=float)
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    y = np.asarray [r['y'] for r in train_rows]
    lower = [0.0, 0.0, 0.001, 0.0, 0.1]
    upper = [50.0, 1.0, 20.0, 20.0, 1000.0]
    best = None
    for tf in (0.3, 1.0, 3.0):
        for ts in (5.0, 20.0, 60.0):
            result = least_squares(lambda p: _evaluate(p, train_rows) - y, [6.0, 0.7, tf, 1.0, ts], bounds=(lower, upper), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, [row])[0])