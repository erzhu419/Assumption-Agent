import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    sink = s / (1.0 + s)
    base = u / (1.2 + u)

    def residual(p):
        vmax, rho, fast, amp, gap = p
        slow = fast + gap
        rise = -np.expm1(-h / fast)
        adapt = -np.expm1(-h / slow)
        return vmax * (1.0 - rho * sink) * base * rise - amp * u * adapt - y

    bounds = ([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf])
    starts = ([5.0, 0.7, 0.15, 2.0, 10.0], [5.0, 0.7, 0.35, 3.0, 25.0], [5.0, 0.7, 0.08, 2.0, 4.0])
    best = min((least_squares(residual, p, bounds=bounds, x_scale='jac', max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11) for p in starts), key=lambda result: result.cost)
    vmax, rho, fast, amp, gap = map(float, best.x)
    _params = (vmax, rho, fast, amp, fast + gap)

def predict(row) -> float:
    vmax, rho, fast, amp, slow = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    rise = -math.expm1(-h / fast)
    adapt = -math.expm1(-h / slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * rise - amp * u * adapt
