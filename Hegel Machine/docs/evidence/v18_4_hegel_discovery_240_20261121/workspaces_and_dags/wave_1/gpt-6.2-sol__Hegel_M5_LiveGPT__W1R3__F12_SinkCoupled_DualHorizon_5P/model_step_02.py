import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _response(u, s, h, p):
    vmax, rho, tau_fast, amp_adapt, tau_gap = p
    tau_slow = tau_fast + tau_gap
    gain = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return gain * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))


def fit(train_rows):
    global _params
    u, s, h, y = np.array([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        return _response(u, s, h, p) - y

    bounds = ([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf])
    best = None
    for start in ([7.0, 0.7, 0.5, 2.0, 10.0], [5.0, 0.5, 0.2, 1.5, 4.0], [9.0, 0.75, 1.0, 3.0, 20.0]):
        result = least_squares(residual, start, bounds=bounds, x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=1500)
        if best is None or result.cost < best.cost:
            best = result
        if math.sqrt(2.0 * best.cost / y.size) < 1e-6:
            break
    _params = best.x


def predict(row):
    return float(_response(row['u_now'], row['s_grad'], row['H'], _params))
