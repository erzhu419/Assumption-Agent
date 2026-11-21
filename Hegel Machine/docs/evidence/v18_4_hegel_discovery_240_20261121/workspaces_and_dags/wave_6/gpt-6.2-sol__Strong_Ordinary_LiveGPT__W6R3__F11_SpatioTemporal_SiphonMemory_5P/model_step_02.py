import math
import numpy as np
from scipy.optimize import least_squares

_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _inputs(rows):
    return np.array([[r[k] for k in _keys] for r in rows], dtype=float).T

def _forecast(p, data):
    a, b, tau, vmax, rho = p
    u, x, z, s, H = data
    un = u / 2.0
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    n = max(1, math.ceil(float(np.max(H)) / 0.004))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + flux, (0.52 * x - z) / tau

    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5*h*k1x, z + 0.5*h*k1z)
        k3x, k3z = rhs(x + 0.5*h*k2x, z + 0.5*h*k2z)
        k4x, k4z = rhs(x + h*k3x, z + h*k3z)
        x = x + (h/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        z = z + (h/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _inputs(rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _forecast(p, data) - y,
                           x0=[0.8, 1.0, 0.3, 3.0, 0.7],
                           bounds=([0.0, 0.0, 0.01, 0.0, 0.0],
                                   [np.inf, np.inf, np.inf, np.inf, 1.0]),
                           x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x

def predict(row) -> float:
    return float(_forecast(_params, _inputs([row]))[0])
