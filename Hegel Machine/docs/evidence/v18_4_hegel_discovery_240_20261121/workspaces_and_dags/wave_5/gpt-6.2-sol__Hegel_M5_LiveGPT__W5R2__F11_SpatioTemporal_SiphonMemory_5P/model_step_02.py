import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _inputs(rows):
    return np.asarray([[r['u_now'], r['x_now'], r['mem_conv'], r['s_grad'], r['H']] for r in rows], dtype=float).T


def _forecast(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = data
    un = u / 0.005
    j = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    n = max(1, math.ceil(float(np.max(H)) / 0.008))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + j, (0.52 * x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5*h*kx, z + 0.5*h*kz)
        mx, mz = rhs(x + 0.5*h*lx, z + 0.5*h*lz)
        nx, nz = rhs(x + h*mx, z + h*mz)
        x = x + (h/6.0)*(kx + 2*lx + 2*mx + nx)
        z = z + (h/6.0)*(kz + 2*lz + 2*mz + nz)
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _inputs(rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _forecast(p, data) - y,
                           [2.0, 1.0, 6.5, 0.70, 0.4],
                           bounds=([0.0, 0.0, 0.0, 0.0, 0.02],
                                   [30.0, 30.0, 40.0, 1.0, 5.0]),
                           x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10)
    _params = result.x


def predict(row):
    return float(_forecast(_params, _inputs([row]))[0])
