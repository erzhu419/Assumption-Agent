import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    best = None
    # Multiple sensible positive starts help resolve the fast/slow timescale fit.
    for tf0 in (0.15, 0.5, 1.0, 2.0):
        for ts0 in (4.0, 8.0, 15.0, 30.0):
            if ts0 <= tf0:
                continue
            p0 = np.array([1.5, 1.5, tf0, ts0], dtype=float)
            lo = np.array([0.0, 0.0, 1e-3, 1e-3])
            hi = np.array([20.0, 20.0, 100.0, 200.0])
            def residual(p):
                return np.asarray([_model(r, p) for r in rows]) - y
            result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=20000)
            p = result.x
            if p[3] <= p[2]:
                continue
            score = float(np.dot(residual(p), residual(p)))
            if best is None or score < best[0]:
                best = (score, p.copy())
    if best is None:
        raise RuntimeError('Could not fit ordered dual-horizon parameters')
    _PARAMS = best[1]
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _PARAMS))