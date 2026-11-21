import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _inputs(row):
    return tuple(float(row[k]) for k in ('u_now', 's_grad', 'H'))


def _evaluate(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, gap = p
    tau_slow = tau_fast + gap
    gain = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return gain * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))


def fit(train_rows):
    global _params
    u, s, h = np.asarray([_inputs(row) for row in train_rows], dtype=float).T
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    bounds = ([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf])
    best = None
    for fast, slow in ((0.3, 8.0), (0.12, 4.0), (0.65, 18.0)):
        result = least_squares(lambda p: _evaluate(p, u, s, h) - y,
                               [6.5, 0.7, fast, 1.5, slow-fast],
                               bounds=bounds, x_scale='jac', max_nfev=1500)
        if best is None or result.cost < best.cost:
            best = result
        if math.sqrt(2.0 * best.cost / y.size) < 1e-6:
            break
    _params = best.x
    return _params


def predict(row):
    return float(_evaluate(_params, *_inputs(row)))