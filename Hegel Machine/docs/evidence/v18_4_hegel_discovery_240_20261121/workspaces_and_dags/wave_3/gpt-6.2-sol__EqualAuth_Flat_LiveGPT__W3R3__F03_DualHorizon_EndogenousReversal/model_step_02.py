import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u, h, y = np.array([[r['u_now'], r['H'], r['y']] for r in rows], dtype=float).T

    def residual(q):
        a_fast, a_slow, tau_fast, gap = np.exp(q)
        tau_slow = tau_fast + gap
        return u * (-a_fast * np.expm1(-h / tau_fast) + a_slow * np.expm1(-h / tau_slow)) - y

    best = None
    best_error = np.inf
    for start in ((2.5, 3.5, 0.6, 8.0), (1.5, 2.5, 0.3, 4.0), (4.0, 6.0, 1.2, 18.0)):
        result = least_squares(residual, np.log(start), bounds=(-14.0, 14.0), ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=500)
        error = float(np.dot(result.fun, result.fun))
        if error < best_error:
            best, best_error = result.x, error
        if best_error <= len(y) * 1e-14:
            break
    a_fast, a_slow, tau_fast, gap = map(float, np.exp(best))
    _params = (a_fast, a_slow, tau_fast, tau_fast + gap)

def predict(row) -> float:
    a_fast, a_slow, tau_fast, tau_slow = _params
    h = float(row['H'])
    return float(row['u_now']) * (-a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / tau_slow))
