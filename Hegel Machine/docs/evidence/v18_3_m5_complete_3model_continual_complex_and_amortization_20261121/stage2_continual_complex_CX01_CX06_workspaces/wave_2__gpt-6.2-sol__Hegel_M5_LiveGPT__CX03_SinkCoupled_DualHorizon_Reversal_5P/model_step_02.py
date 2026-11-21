import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    rows = list(train_rows)
    u, s, h, y = np.array([[r[k] for k in ('u_now', 's_grad', 'H', 'y')] for r in rows], dtype=float).T
    sink = s / (1.0 + s)
    drive = u / (1.2 + u)

    def residual(p):
        vmax, rho, tau_fast, amp_adapt, tau_slow = p
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return vmax * (1.0 - rho * sink) * drive * fast - amp_adapt * u * slow - y

    bounds = ([0.0, 0.0, 1e-5, 0.0, 1e-5], [np.inf, 1.0, np.inf, np.inf, np.inf])
    best = None
    for tf, ts in ((0.3, 12.0), (0.1, 5.0), (0.9, 25.0), (0.5, 60.0)):
        result = least_squares(residual, [6.5, 0.70, tf, 2.0, ts], bounds=bounds, x_scale='jac', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x


def predict(row) -> float:
    vmax, rho, tau_fast, amp_adapt, tau_slow = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / tau_slow)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow)
