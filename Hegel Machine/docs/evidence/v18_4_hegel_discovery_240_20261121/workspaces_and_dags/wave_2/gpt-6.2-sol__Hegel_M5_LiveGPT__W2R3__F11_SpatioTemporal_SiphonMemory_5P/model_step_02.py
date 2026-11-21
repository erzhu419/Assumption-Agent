import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _read(rows):
    return np.asarray([[row[k] for k in _KEYS] for row in rows], dtype=float).T


def _evolve(data, p, n):
    u, x, z, s, H = data
    a, b, vmax, rho, tau = p
    un = u / 0.05
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    dt = H / n

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5 * dt * kx, z + 0.5 * dt * kz)
        mx, mz = rhs(x + 0.5 * dt * lx, z + 0.5 * dt * lz)
        nx, nz = rhs(x + dt * mx, z + dt * mz)
        x = x + (dt / 6.0) * (kx + 2.0 * lx + 2.0 * mx + nx)
        z = z + (dt / 6.0) * (kz + 2.0 * lz + 2.0 * mz + nz)
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _read(rows)
    y = np.asarray([row['y'] for row in rows], dtype=float)
    n = max(1, math.ceil(float(np.max(data[-1])) / 0.005))
    result = least_squares(
        lambda p: _evolve(data, p, n) - y,
        x0=[1.5, 1.0, 6.5, 0.70, 0.5],
        bounds=([0.0, 0.0, 0.0, 0.0, 0.01], [30.0, 30.0, np.inf, 1.0, np.inf]),
        x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    _params = result.x
    return _params


def predict(row) -> float:
    data = _read([row])
    n = max(1, math.ceil(float(row['H']) / 0.005))
    return float(_evolve(data, _params, n)[0])
