import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - ys
    lo = np.array([0.01, 0.0, 0.005, 0.0, 0.05])
    hi = np.array([30.0, 0.999, 10.0, 10.0, 300.0])
    starts = [
        [6.0, 0.70, tf, a, ts]
        for tf in (0.05, 0.3, 1.0, 3.0)
        for a in (0.1, 0.5, 1.0, 2.0)
        for ts in (8.0, 20.0, 50.0, 120.0)
    ]
    best = None
    for x0 in starts:
        res = least_squares(residual, np.clip(x0, lo + 1e-8, hi - 1e-8),
                            bounds=(lo, hi), max_nfev=5000, xtol=1e-12,
                            ftol=1e-12, gtol=1e-12)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _PARAMS = np.asarray(best[1], dtype=float)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _PARAMS))