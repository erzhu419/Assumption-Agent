import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _inputs(rows):
    return np.array([[float(row[k]) for k in _KEYS] for row in rows], dtype=float).T


def _endpoint(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = data
    un = u / 0.05
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    n = np.maximum(1, np.ceil(H / 0.004).astype(int))
    dt = H / n

    def rhs(x, z):
        return -a * x + b * z + flux, (0.52 * x - z) / tau

    for i in range(int(n.max())):
        d = np.where(i < n, dt, 0.0)
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*d*k1, z + 0.5*d*l1)
        k3, l3 = rhs(x + 0.5*d*k2, z + 0.5*d*l2)
        k4, l4 = rhs(x + d*k3, z + d*l3)
        x = x + d*(k1 + 2*k2 + 2*k3 + k4)/6.0
        z = z + d*(l1 + 2*l2 + 2*l3 + l4)/6.0
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _inputs(rows)
    y = np.array([float(row['y']) for row in rows])
    result = least_squares(lambda p: _endpoint(p, data) - y,
                           x0=[1.0, 1.0, 4.0, 0.7, 0.3],
                           bounds=([0.0, 0.0, 0.0, 0.0, 0.002],
                                   [math.inf, math.inf, math.inf, 1.0, math.inf]),
                           x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params


def predict(row) -> float:
    return float(_endpoint(_params, _inputs([row]))[0])
