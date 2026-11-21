import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')

def _read(rows):
    return np.array([[float(row[k]) for k in _keys] for row in rows], dtype=float).T

def _endpoint(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, horizon = data
    u = u / 2.0
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(48, math.ceil(float(np.max(horizon)) / (0.5 * tau)))
    h = horizon / n

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * h * k1, z + 0.5 * h * l1)
        k3, l3 = rhs(x + 0.5 * h * k2, z + 0.5 * h * l2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x = x + h * (k1 + 2.0 * (k2 + k3) + k4) / 6.0
        z = z + h * (l1 + 2.0 * (l2 + l3) + l4) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _read(rows)
    y = np.array([float(row['y']) for row in rows])
    best = None
    for tau0 in (0.3, 0.08, 0.9):
        result = least_squares(lambda p: _endpoint(p, data) - y,
                               [0.8, 0.7, 6.5, 0.70, tau0],
                               bounds=([0.0, 0.0, 0.0, 0.0, 0.005],
                                       [np.inf, np.inf, np.inf, 1.0, np.inf]),
                               x_scale='jac', max_nfev=250)
        if best is None or result.cost < best.cost:
            best = result
        if np.linalg.norm(best.fun) <= 1e-6 * math.sqrt(len(rows)):
            break
    _params = best.x
    return _params.copy()

def predict(row) -> float:
    return float(_endpoint(_params, _read([row]))[0])
