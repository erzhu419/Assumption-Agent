import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    data = np.asarray([[r[k] for k in ('u_now', 's_grad', 'H', 'y')] for r in train_rows], dtype=float)
    u, s, h, y = data.T
    def residual(p):
        return _response(p, u, s, h) - y
    bounds = ([0., 0., 1e-5, 0., 1e-5], [np.inf, 1., np.inf, np.inf, np.inf])
    best = None
    for tf, ts in ((0.3, 12.), (0.1, 5.), (0.7, 25.), (1.5, 50.)):
        result = least_squares(residual, [6., 0.6, tf, 1.5, ts], bounds=bounds, x_scale='jac', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x.copy()
    return _params

def predict(row) -> float:
    v, rho, tf, a, ts = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    return float(v * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * (-math.expm1(-h / tf)) - a * u * (-math.expm1(-h / ts)))