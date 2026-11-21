import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
    adaptation = amp_adapt * u * (1.0 - math.exp(-H / tau_slow))
    return base - adaptation

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    lower = np.array([0.0, 0.0, 0.0, 1e-3, 1e-3])
    upper = np.array([30.0, 0.999, 10.0, 100.0, 300.0])
    starts = [
        [6.0, 0.70, 0.20, 1.0, 10.0],
        [6.0, 0.70, 0.20, 3.0, 30.0],
        [6.0, 0.70, 0.30, 0.5, 20.0],
        [4.0, 0.50, 0.15, 2.0, 50.0],
    ]
    best = None
    for start in starts:
        result = least_squares(
            lambda p: np.asarray([_value(r, p) for r in rows]) - y,
            x0=np.asarray(start, dtype=float), bounds=(lower, upper),
            max_nfev=30000, xtol=1e-13, ftol=1e-13, gtol=1e-13
        )
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = np.asarray(best[1], dtype=float)
    return {name: float(v) for name, v in zip(
        ('vmax', 'rho', 'amp_adapt', 'tau_fast', 'tau_slow'), _PARAMS
    )}

def predict(row) -> float:
    if _PARAMS is None:
        raise ValueError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _PARAMS))