import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    siphon = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return siphon * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_evaluate(p, r) for r in rows]) - y
    best = None
    for tf in (0.2, 0.5, 1.0, 2.0):
        for ts in (4.0, 10.0, 20.0, 40.0):
            result = least_squares(residual, [6.0, 0.7, tf, 1.0, ts], bounds=([0.0, 0.0, 0.001, 0.0, 0.1], [1000.0, 1.5, 100.0, 100.0, 1000.0]), max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, row))