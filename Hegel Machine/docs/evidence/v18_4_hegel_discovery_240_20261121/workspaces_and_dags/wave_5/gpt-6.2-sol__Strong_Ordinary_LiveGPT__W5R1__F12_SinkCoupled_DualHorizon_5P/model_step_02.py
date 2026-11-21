import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float)
    u, s, h, y = data.T
    sat = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, fast, amp, gap = p
        slow = fast + gap
        rise = -np.expm1(-h / fast)
        adapt = -np.expm1(-h / slow)
        return vmax * (1.0 - rho * sink) * sat * rise - amp * u * adapt - y

    bounds = ([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf])
    best = None
    for fast, gap in ((0.2, 8.0), (0.08, 3.0), (0.5, 20.0), (0.15, 50.0)):
        result = least_squares(residual, [6.0, 0.7, fast, 1.5, gap], bounds=bounds, x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=1200)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x


def predict(row):
    vmax, rho, fast, amp, gap = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    rise = -math.expm1(-h / fast)
    adapt = -math.expm1(-h / (fast + gap))
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * rise - amp * u * adapt)
