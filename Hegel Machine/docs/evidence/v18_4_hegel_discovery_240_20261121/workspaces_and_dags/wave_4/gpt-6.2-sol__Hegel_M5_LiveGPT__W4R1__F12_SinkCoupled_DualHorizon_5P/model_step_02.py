import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, h, y = np.array([[r[k] for k in ('u_now', 's_grad', 'H', 'y')] for r in train_rows], dtype=float).T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, tf, amp, gap = p
        fast = -np.expm1(-h / tf)
        slow = -np.expm1(-h / (tf + gap))
        return vmax * (1.0 - rho * sink) * drive * fast - amp * u * slow - y

    best = None
    for start in ([6.5, 0.70, 0.30, 1.5, 12.0], [8.0, 0.70, 0.15, 2.0, 25.0], [5.0, 0.65, 0.55, 1.0, 5.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf]), x_scale='jac', max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    vmax, rho, tf, amp, gap = best.x
    _params = (vmax, rho, tf, amp, tf + gap)
    return _params

def predict(row) -> float:
    vmax, rho, tf, amp, ts = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    fast = -math.expm1(-h / tf)
    slow = -math.expm1(-h / ts)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp * u * slow)
