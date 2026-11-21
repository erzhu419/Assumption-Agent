import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _rows(rows):
    return np.array([[r[k] for k in _KEYS] for r in rows], dtype=float)


def _forecast(p, data):
    a, b, tau, vmax, rho = p
    u, x, z, s, H = data.T
    u = u / 0.05
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(H)) / 0.004))
    h = H / n
    x, z = x.copy(), z.copy()

    def rhs(x, z):
        return flux - a*x + b*z, (0.52*x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5*h*kx, z + 0.5*h*kz)
        mx, mz = rhs(x + 0.5*h*lx, z + 0.5*h*lz)
        nx, nz = rhs(x + h*mx, z + h*mz)
        x += (h/6.0) * (kx + 2*lx + 2*mx + nx)
        z += (h/6.0) * (kz + 2*lz + 2*mz + nz)
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _rows(rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _forecast(p, data) - y,
                           [1.0, 1.0, 0.3, 5.0, 0.7],
                           bounds=([0, 0, 0.002, 0, 0],
                                   [np.inf, np.inf, np.inf, np.inf, 1]),
                           x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params


def predict(row) -> float:
    return float(_forecast(_params, _rows([row]))[0])
