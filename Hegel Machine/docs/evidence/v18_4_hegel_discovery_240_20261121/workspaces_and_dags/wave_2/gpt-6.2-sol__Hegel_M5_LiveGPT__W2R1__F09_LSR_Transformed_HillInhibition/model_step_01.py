import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 2.5
_params = None

def _law(u, s, p):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([float(r['u_now']) / _TAU for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    scale = max(1.0, float(np.max(np.abs(y))))
    def residual(p):
        return _law(u, s, p) - y
    bounds = ([0.0, 1e-10, 0.0, 0.0], [math.inf] * 4)
    best = None
    for k, a in [(1.0, 0.7), (0.2, 0.2), (4.0, 1.5)]:
        result = least_squares(residual, [2.0 * scale, k, a, 0.1], bounds=bounds, x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x
    return _params

def predict(row) -> float:
    return float(_law(float(row['u_now']) / _TAU, float(row['s_grad']), _params))