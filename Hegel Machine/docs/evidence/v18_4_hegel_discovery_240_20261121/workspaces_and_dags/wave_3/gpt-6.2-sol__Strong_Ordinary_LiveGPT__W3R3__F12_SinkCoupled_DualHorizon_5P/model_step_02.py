import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, h, y = np.array([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    sink = s / (1.0 + s)
    base = u / (1.2 + u)

    def residual(p):
        vmax, rho, tf, amp, ts = p
        fast = -np.expm1(-h / tf)
        slow = -np.expm1(-h / ts)
        return vmax * (1.0 - rho * sink) * base * fast - amp * u * slow - y

    bounds = ([0.0, 0.0, 0.001, 0.0, 0.5], [np.inf, 1.0, 2.0, np.inf, np.inf])
    best = None
    for start in ([5.0, 0.7, 0.25, 2.0, 12.0], [8.0, 0.5, 0.5, 3.0, 25.0], [4.0, 0.8, 0.1, 1.5, 5.0]):
        result = least_squares(residual, start, bounds=bounds, x_scale='jac', max_nfev=600, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x

def predict(row):
    vmax, rho, tf, amp, ts = _params
    u, s, h = row['u_now'], row['s_grad'], row['H']
    fast = -math.expm1(-h / tf)
    slow = -math.expm1(-h / ts)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp * u * slow)
