import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _forecast(p, rows):
    a, b, tau, vmax, rho = p
    u, x, z, g, horizon = rows.T
    un = u / 0.05
    source = vmax * (1.0 - rho * g / (1.0 + g)) * un / (1.2 + un)
    n = max(1, math.ceil(float(np.max(horizon)) * max(100.0, 4.0 / tau, 4.0 * a)))
    h = horizon / n

    def rhs(x, z):
        return -a * x + b * z + source, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        z = z + (h/6.0)*(l1 + 2*l2 + 2*l3 + l4)
    return x


def fit(train_rows):
    global _params
    rows = np.array([[r[k] for k in _KEYS] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, 1.0])
    best = None
    for tau0 in (0.05, 0.16, 0.45):
        result = least_squares(lambda p: _forecast(p, rows) - y,
                               [0.7, 1.2, tau0, 3.0, 0.5], bounds=bounds,
                               x_scale='jac', max_nfev=250)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-10:
            break
    _params = best.x
    return _params


def predict(row) -> float:
    rows = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_forecast(_params, rows)[0])