import math
import numpy as np
from scipy.optimize import least_squares


def _forecast(row, params):
    a, b, c, tau = params
    x = float(row['x_now'])
    z = row.get('mem_conv')
    if z is None:
        prefix = sorted(row['raw_prefix'], key=lambda s: s['t'])
        z = 0.52 * prefix[0]['x']
        for left, right in zip(prefix, prefix[1:]):
            q = (right['t'] - left['t']) / tau
            decay = math.exp(-q)
            z = decay * z + 0.26 * q * (decay * left['x'] + right['x'])
    drive = c * float(row['u_now'])
    H = float(row['H'])
    n = max(1, math.ceil(H / 0.004))
    h = H / n

    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5*h*kx, z + 0.5*h*kz)
        mx, mz = rhs(x + 0.5*h*lx, z + 0.5*h*lz)
        nx, nz = rhs(x + h*mx, z + h*mz)
        x += h/6 * (kx + 2*lx + 2*mx + nx)
        z += h/6 * (kz + 2*lz + 2*mz + nz)
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)

    def residual(p):
        return np.asarray([_forecast(r, p) for r in rows]) - y

    _params = least_squares(residual, [2.0, 1.0, 1.0, 0.2], bounds=([0, 0, 0, 0.02], [np.inf]*4), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return _params


def predict(row) -> float:
    return _forecast(row, _params)
