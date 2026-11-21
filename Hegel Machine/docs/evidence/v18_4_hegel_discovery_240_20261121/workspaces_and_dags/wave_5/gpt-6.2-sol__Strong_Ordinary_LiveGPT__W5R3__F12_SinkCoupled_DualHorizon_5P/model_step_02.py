import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, h, y = np.array([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    sat = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, tau_fast, amp, tau_slow = p
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return vmax * (1.0 - rho * sink) * sat * fast - amp * u * slow - y

    bounds = ([0.0, 0.0, 1e-5, 0.0, 1e-5], [np.inf, 1.0, np.inf, np.inf, np.inf])
    best = None
    for start in ([6.0, 0.7, 0.25, 1.5, 12.0], [5.0, 0.5, 0.6, 2.0, 25.0], [8.0, 0.8, 0.12, 1.2, 6.0]):
        result = least_squares(residual, start, bounds=bounds, x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-13:
            break
    _params = tuple(map(float, best.x))
    return _params

def predict(row) -> float:
    vmax, rho, tau_fast, amp, tau_slow = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / tau_slow)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp * u * slow)
