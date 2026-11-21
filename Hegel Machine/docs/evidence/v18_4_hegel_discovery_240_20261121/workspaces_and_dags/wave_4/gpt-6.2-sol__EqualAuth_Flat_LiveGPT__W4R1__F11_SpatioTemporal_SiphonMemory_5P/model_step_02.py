import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _inputs(rows):
    return np.asarray([[row[k] for k in _KEYS] for row in rows], dtype=float).T

def _forecast(params, inputs):
    a, b, tau, vmax, rho = params
    u, x, z, s, H = inputs
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(64, math.ceil(8.0 * float(np.max(H)) * max(a, 1.0 / tau)))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * h * k1, z + 0.5 * h * l1)
        k3, l3 = rhs(x + 0.5 * h * k2, z + 0.5 * h * l2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x = x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        z = z + h * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    inputs = _inputs(rows)
    target = np.asarray([row['y'] for row in rows], dtype=float)
    _params = least_squares(
        lambda p: _forecast(p, inputs) - target,
        [0.7, 0.8, 0.35, 3.0, 0.5],
        bounds=([0.0, 0.0, 0.02, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, 1.0]),
        ftol=1e-11, xtol=1e-11, gtol=1e-11,
    ).x

def predict(row):
    return float(_forecast(_params, _inputs([row]))[0])