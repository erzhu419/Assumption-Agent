import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _inputs(rows):
    return np.asarray([[float(row[k]) for k in _KEYS] for row in rows], dtype=float).T


def _forecast(p, data):
    a, b, tau, vmax, rho = p
    u, x, z, s, H = data
    un = u / 2.0
    source = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    n = max(1, math.ceil(float(np.max(H)) / 0.01))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + source, (0.52 * x - z) / tau

    for _ in range(n):
        kx1, kz1 = rhs(x, z)
        kx2, kz2 = rhs(x + 0.5*h*kx1, z + 0.5*h*kz1)
        kx3, kz3 = rhs(x + 0.5*h*kx2, z + 0.5*h*kz2)
        kx4, kz4 = rhs(x + h*kx3, z + h*kz3)
        x = x + (h/6.0)*(kx1 + 2*kx2 + 2*kx3 + kx4)
        z = z + (h/6.0)*(kz1 + 2*kz2 + 2*kz3 + kz4)
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _inputs(rows)
    y = np.asarray([float(row['y']) for row in rows])
    bounds = ([0.0, 0.0, 0.01, 0.0, 0.0], [30.0, np.inf, np.inf, np.inf, 1.0])

    def residuals(p):
        return _forecast(p, data) - y

    best = None
    for tau0 in (0.25, 1.0, 2.0):
        result = least_squares(residuals, [0.8, 1.0, tau0, 3.0, 0.5], bounds=bounds, x_scale='jac', max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x
    return _params


def predict(row) -> float:
    return float(_forecast(_params, _inputs([row]))[0])
