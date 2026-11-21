import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _inputs(rows):
    return np.asarray([[float(row[k]) for k in _KEYS] for row in rows], dtype=float)


def _forecast(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, horizon = data.T
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    steps = max(64, math.ceil(4.0 * np.max(np.abs(horizon)) * max(a, b, 1.0 / tau)))
    dt = horizon / steps

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(steps):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * dt * k1, z + 0.5 * dt * l1)
        k3, l3 = rhs(x + 0.5 * dt * k2, z + 0.5 * dt * l2)
        k4, l4 = rhs(x + dt * k3, z + dt * l3)
        x = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        z = z + dt * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _inputs(rows)
    y = np.asarray([float(row['y']) for row in rows])
    bounds = ([0.0, 0.0, 1.0, 0.05, 0.02], [20.0, 20.0, 15.0, 0.95, 20.0])
    starts = [(1.0, 0.8, 6.5, 0.70, 0.45), (1.7, 1.5, 8.0, 0.70, 0.15), (0.6, 0.6, 5.0, 0.70, 1.5)]
    best = None
    for start in starts:
        result = least_squares(lambda p: _forecast(p, data) - y, start, bounds=bounds, x_scale='jac', max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if result.cost < 1e-12:
            break
    _params = best.x
    return _params


def predict(row) -> float:
    return float(_forecast(_params, _inputs([row]))[0])