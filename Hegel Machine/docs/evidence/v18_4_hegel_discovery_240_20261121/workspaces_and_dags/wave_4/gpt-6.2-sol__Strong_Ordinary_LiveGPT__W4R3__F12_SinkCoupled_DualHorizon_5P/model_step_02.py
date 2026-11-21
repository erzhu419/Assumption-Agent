import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    base = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, amp, tf, gap = p
        ts = tf + gap
        return vmax * (1.0 - rho * sink) * base * (-np.expm1(-h / tf)) - amp * u * (-np.expm1(-h / ts)) - y

    bounds = ([0.0, 0.0, 0.0, 1e-5, 1e-5], [np.inf, 1.0, np.inf, np.inf, np.inf])
    best = None
    for tf, ts in ((0.2, 12.0), (0.06, 5.0), (0.6, 30.0)):
        result = least_squares(residual, [6.0, 0.7, 2.0, tf, ts - tf], bounds=bounds, x_scale='jac', max_nfev=800)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    vmax, rho, amp, tf, gap = best.x
    _params = (vmax, rho, amp, tf, tf + gap)
    return _params

def predict(row) -> float:
    vmax, rho, amp, tf, ts = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return float(base * (-math.expm1(-h / tf)) - amp * u * (-math.expm1(-h / ts)))
