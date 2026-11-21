import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(rows, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.asarray([r['u_now'] for r in rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in rows], dtype=float)
    h = np.asarray([r['H'] for r in rows], dtype=float)
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            * (-np.expm1(-h / tau_fast))
            - amp_adapt * u * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    lower = [0.0, 0.0, 0.001, 0.0, 0.1]
    upper = [100.0, 1.5, 20.0, 10.0, 1000.0]
    best = None
    for tf in (0.15, 0.5, 1.5):
        for ts in (5.0, 20.0, 80.0):
            result = least_squares(lambda p: _values(rows, p) - y,
                                   [5.0, 0.7, tf, 1.0, ts],
                                   bounds=(lower, upper), max_nfev=4000,
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or result.cost < best.cost:
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_values([row], _params)[0])