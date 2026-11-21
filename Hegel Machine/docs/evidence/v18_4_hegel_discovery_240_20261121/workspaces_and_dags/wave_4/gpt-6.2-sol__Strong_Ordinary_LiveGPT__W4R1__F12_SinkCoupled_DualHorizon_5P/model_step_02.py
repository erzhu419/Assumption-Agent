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
        vmax, rho, tau_fast, amp, tau_slow = p
        return vmax * (1.0 - rho * sink) * drive * (-np.expm1(-h / tau_fast)) - amp * u * (-np.expm1(-h / tau_slow)) - y

    bounds = ([0.0, 0.0, 1e-5, 0.0, 1e-5], [np.inf, 1.0, np.inf, np.inf, np.inf])
    best = None
    for fast, slow in ((0.25, 12.0), (0.08, 5.0), (0.65, 35.0)):
        result = least_squares(residual, [5.0, 0.7, fast, 1.4, slow], bounds=bounds, x_scale='jac', max_nfev=600, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x

def predict(row) -> float:
    vmax, rho, tau_fast, amp, tau_slow = _params
    u, s, h = row['u_now'], row['s_grad'], row['H']
    fast = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * (-math.expm1(-h / tau_fast))
    slow = amp * u * (-math.expm1(-h / tau_slow))
    return float(fast - slow)
