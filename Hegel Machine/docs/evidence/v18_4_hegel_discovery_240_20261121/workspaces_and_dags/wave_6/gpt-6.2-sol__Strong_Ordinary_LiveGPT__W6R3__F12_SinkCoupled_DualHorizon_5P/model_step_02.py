import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(p):
        return _response(p, u, s, h) - y
    bounds = ([0.0, 0.0, 1e-6, 0.0, 1e-6], [math.inf, 1.0, math.inf, math.inf, math.inf])
    best = None
    for start in [(7.0, 0.7, 0.25, 2.0, 12.0), (5.0, 0.5, 0.10, 1.5, 5.0), (10.0, 0.85, 0.6, 3.0, 25.0)]:
        result = least_squares(residual, start, bounds=bounds, x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=600)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x

def predict(row):
    return float(_response(_params, row['u_now'], row['s_grad'], row['H']))