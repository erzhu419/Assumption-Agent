import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')


def _inputs(rows):
    return np.array([[row[k] for k in _keys] for row in rows], dtype=float).T


def _endpoint(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = data
    un = u / 2.0
    j = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    n = max(1, math.ceil(float(np.max(H)) / 0.0075))
    h = H / n

    def step(x, z):
        return h * (-a*x + b*z + j), (h/tau) * (0.52*x - z)

    for _ in range(n):
        k1, l1 = step(x, z)
        k2, l2 = step(x + k1/2, z + l1/2)
        k3, l3 = step(x + k2/2, z + l2/2)
        k4, l4 = step(x + k3, z + l3)
        x = x + (k1 + 2*k2 + 2*k3 + k4)/6
        z = z + (l1 + 2*l2 + 2*l3 + l4)/6
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _inputs(rows)
    target = np.array([row['y'] for row in rows], dtype=float)
    result = least_squares(lambda p: _endpoint(p, data) - target,
                           [1.0, 1.0, 5.0, 0.5, 0.5],
                           bounds=([0, 0, 0, 0, 0.015],
                                   [np.inf, np.inf, np.inf, 1, np.inf]),
                           x_scale='jac', max_nfev=200)
    _params = result.x
    return _params


def predict(row):
    return float(_endpoint(_params, _inputs([row]))[0])
