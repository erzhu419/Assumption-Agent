import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _inputs(rows):
    return np.asarray([[r['u_now'], r['x_now'], r['mem_conv'], r['s_grad'], r['H']] for r in rows], dtype=float).T

def _forecast(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = data
    force = vmax * u / (1.2 + u) * (1.0 - rho * s / (1.0 + s))
    n = max(1, math.ceil(float(np.max(H)) / 0.004))
    h = H / n

    def rhs(x, z):
        return -a*x + b*z + force, (0.52*x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x = x + h*(k1 + 2*k2 + 2*k3 + k4)/6
        z = z + h*(l1 + 2*l2 + 2*l3 + l4)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _inputs(rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _forecast(p, data) - y,
                           [1.5, 1.0, 6.0, 0.7, 0.5],
                           bounds=([0, 0, 0, 0, 0.01], [30, 30, np.inf, 1, np.inf]))
    _params = result.x
    return _params

def predict(row):
    return float(_forecast(_params, _inputs([row]))[0])