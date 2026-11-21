import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _inputs(rows):
    return np.asarray([[float(row[key]) for key in _KEYS] for row in rows], dtype=float).T


def _forecast(params, inputs):
    vmax, rho, a, b, tau = params
    u, x, z, s, H = inputs
    u = u / 2.0
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(H)) / 0.004))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + J, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x = x + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
        z = z + h*(l1 + 2*l2 + 2*l3 + l4)/6.0
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    inputs = _inputs(rows)
    y = np.asarray([float(row['y']) for row in rows])
    result = least_squares(lambda p: _forecast(p, inputs) - y,
                           [5.0, 0.7, 1.3, 1.4, 0.2],
                           bounds=([0, 0, 0, 0, 0.002], [np.inf, 1, np.inf, np.inf, np.inf]),
                           x_scale='jac')
    _params = result.x
    return _params


def predict(row):
    return float(_forecast(_params, _inputs([row]))[0])
