import math
import numpy as np
from scipy.optimize import least_squares

_DT = 0.004
_params = None


def _inputs(rows):
    u, x, z, s, h = np.array([[r[k] for k in ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')] for r in rows], dtype=float).T
    u = u / 0.05
    return u / (1.2 + u), x, z, s / (1.0 + s), h


def _forecast(p, data):
    a, b, vmax, rho, tau = p
    sat, x, z, sink, h = data
    j = vmax * (1.0 - rho * sink) * sat

    def rhs(x, z):
        return -a * x + b * z + j, (0.52 * x - z) / tau

    for i in range(math.ceil(float(np.max(h)) / _DT)):
        d = np.clip(h - i * _DT, 0.0, _DT)
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * d * k1, z + 0.5 * d * l1)
        k3, l3 = rhs(x + 0.5 * d * k2, z + 0.5 * d * l2)
        k4, l4 = rhs(x + d * k3, z + d * l3)
        x = x + d * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        z = z + d * (l1 + 2*l2 + 2*l3 + l4) / 6.0
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _inputs(rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _forecast(p, data) - y,
                           [0.8, 1.6, 7.0, 0.7, 0.08],
                           bounds=([0.0, 0.0, 0.0, 0.0, 0.002],
                                   [np.inf, np.inf, np.inf, 1.0, np.inf]),
                           x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params.copy()


def predict(row):
    return float(_forecast(_params, _inputs([row]))[0])
