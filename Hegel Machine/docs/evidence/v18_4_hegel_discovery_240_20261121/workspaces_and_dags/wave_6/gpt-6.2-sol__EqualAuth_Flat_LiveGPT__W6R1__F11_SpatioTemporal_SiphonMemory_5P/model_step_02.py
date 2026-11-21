import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _features(rows):
    return np.array([[r[k] for k in _KEYS] for r in rows], dtype=float).T


def _forecast(p, features, steps):
    a, b, tau, vmax, rho = p
    u, x, z, s, horizon = features
    un = u / 0.05
    influx = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    dt = horizon / steps

    def rhs(x, z):
        return -a * x + b * z + influx, (0.52 * x - z) / tau

    for _ in range(steps):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * dt * k1, z + 0.5 * dt * l1)
        k3, l3 = rhs(x + 0.5 * dt * k2, z + 0.5 * dt * l2)
        k4, l4 = rhs(x + dt * k3, z + dt * l3)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        z = z + dt * (l1 + 2 * l2 + 2 * l3 + l4) / 6.0
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    features = _features(rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    steps = max(1, math.ceil(float(np.max(features[-1])) / 0.01))
    result = least_squares(lambda p: _forecast(p, features, steps) - y,
                           x0=[1.0, 1.0, 0.3, 4.0, 0.5],
                           bounds=([0.0, 0.0, 0.01, 0.0, 0.0],
                                   [np.inf, np.inf, np.inf, np.inf, 1.0]),
                           x_scale='jac')
    _params = result.x
    return _params


def predict(row):
    steps = max(1, math.ceil(float(row['H']) / 0.01))
    return float(_forecast(_params, _features([row]), steps)[0])