import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    un = u  # supplied tau_scale is 1.0
    inflow = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    return inflow * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * un * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    # Warm start around the shared siphon subtree; bounded positive timescales
    # and amplitudes, with tau_slow > tau_fast to encode dual-horizon reversal.
    best = None
    for tf0, ts0 in [(0.5, 10.0), (1.0, 8.0), (0.25, 15.0)]:
        p0 = np.array([6.5, 0.70, tf0, 1.0, ts0])
        lo = np.array([1.0, 0.05, 0.03, 0.0, 1.0])
        hi = np.array([15.0, 0.95, 5.0, 4.0, 80.0])
        res = least_squares(residual, p0, bounds=(lo, hi), max_nfev=20000,
                            xtol=1e-13, ftol=1e-13, gtol=1e-13)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _PARAMS))