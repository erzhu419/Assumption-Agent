import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(rows, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    siphon = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return siphon * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    y = np.array [r['y'] for r in train_rows], dtype=float)
    lower = [0.0, 0.0, 0.001, 0.0, 0.1]
    upper = [100.0, 2.0, 20.0, 20.0, 500.0]
    best = None
    for tf in (0.15, 0.5, 1.5):
        for ts in (5.0, 20.0, 60.0):
            result = least_squares(lambda p: _values(train_rows, p) - y,
                                   [8.0, 0.7, tf, 1.0, ts],
                                   bounds=(lower, upper), max_nfev=5000,
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    return float(_values([row], _params)[0])