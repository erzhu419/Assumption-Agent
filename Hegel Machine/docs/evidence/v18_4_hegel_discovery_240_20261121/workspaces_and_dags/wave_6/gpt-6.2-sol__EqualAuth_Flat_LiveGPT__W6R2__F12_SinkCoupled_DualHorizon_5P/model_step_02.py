import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, tau_fast, amp, gap = p
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / (tau_fast + gap))
        return vmax * (1.0 - rho * sink) * drive * fast - amp * u * slow - y

    bounds = ([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf])
    starts = ([7.0, 0.6, 0.25, 2.0, 8.0], [10.0, 0.75, 0.6, 3.0, 20.0], [5.0, 0.4, 0.1, 1.5, 3.0])
    best = min((least_squares(residual, p, bounds=bounds, x_scale='jac', max_nfev=1000) for p in starts), key=lambda result: result.cost)
    _params = best.x

def predict(row) -> float:
    vmax, rho, tau_fast, amp, gap = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / (tau_fast + gap))
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp * u * slow)