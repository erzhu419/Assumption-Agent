import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([(r['u_now'], r['s_grad'], r['H'], r['y']) for r in train_rows], dtype=float).T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, tf, amp, ts = p
        fast = -np.expm1(-h / tf)
        slow = -np.expm1(-h / ts)
        return vmax * (1.0 - rho * sink) * drive * fast - amp * u * slow - y

    bounds = ([0.0, 0.0, 1e-5, 0.0, 1e-5], [np.inf, 1.0, np.inf, np.inf, np.inf])
    starts = [(6.5, 0.7, 0.25, 2.0, 8.0), (8.0, 0.65, 0.6, 3.0, 18.0), (4.0, 0.75, 0.1, 1.5, 4.0)]
    fit_results = [least_squares(residual, p, bounds=bounds, x_scale='jac', max_nfev=1500, ftol=1e-12, xtol=1e-12, gtol=1e-12) for p in starts]
    _params = min(fit_results, key=lambda result: result.cost).x
    return _params

def predict(row):
    vmax, rho, tf, amp, ts = _params
    u, s, h = row['u_now'], row['s_grad'], row['H']
    fast = -math.expm1(-h / tf)
    slow = -math.expm1(-h / ts)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp * u * slow)
