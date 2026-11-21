import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    u, s, h, y = np.array([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, fast, amp, gap = p
        rise = -np.expm1(-h / fast)
        adaptation = -np.expm1(-h / (fast + gap))
        return vmax * (1.0 - rho * sink) * drive * rise - amp * u * adaptation - y

    bounds = ([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf])
    best = None
    for fast, slow in [(0.4, 8.0), (0.15, 3.0), (0.8, 20.0)]:
        result = least_squares(residual, [6.5, 0.7, fast, 1.0, slow - fast], bounds=bounds, x_scale='jac', max_nfev=1500)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = tuple(map(float, best.x))


def predict(row) -> float:
    vmax, rho, fast, amp, gap = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    rise = -math.expm1(-h / fast)
    adaptation = -math.expm1(-h / (fast + gap))
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * rise - amp * u * adaptation)
