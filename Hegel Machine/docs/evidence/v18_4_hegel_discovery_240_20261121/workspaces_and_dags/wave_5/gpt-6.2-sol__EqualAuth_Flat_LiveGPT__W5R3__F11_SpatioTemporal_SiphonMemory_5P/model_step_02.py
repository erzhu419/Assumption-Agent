import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _simulate(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, h = data.T
    u = u / 2.0
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    rate = max(a + b, 1.52 / tau)
    n = max(32, math.ceil(4.0 * float(np.max(h)) * rate))
    dt = h / n

    def rhs(x, z):
        return -a * x + b * z + j, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * dt * k1, z + 0.5 * dt * l1)
        k3, l3 = rhs(x + 0.5 * dt * k2, z + 0.5 * dt * l2)
        k4, l4 = rhs(x + dt * k3, z + dt * l3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z = z + (dt / 6.0) * (l1 + 2.0 * l2 + 2.0 * l3 + l4)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.array([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _simulate(p, data) - y,
                           x0=[0.6, 1.4, 3.0, 0.6, 0.25],
                           bounds=([0.0, 0.0, 0.0, 0.0, 0.01],
                                   [np.inf, np.inf, np.inf, 1.0, np.inf]),
                           x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10)
    _params = result.x

def predict(row):
    data = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_simulate(_params, data)[0])