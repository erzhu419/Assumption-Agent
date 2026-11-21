import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _forecast(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * prefix[0]['x']
        for left, right in zip(prefix, prefix[1:]):
            r = (right['t'] - left['t']) / tau
            decay = math.exp(-r)
            z = decay * z + 0.26 * r * (decay * left['x'] + right['x'])
    drive = c * float(row['u_now'])
    H = float(row['H'])
    n = max(1, math.ceil(H / 0.004))
    dt = H / n

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        kx1, kz1 = rhs(x, z)
        kx2, kz2 = rhs(x + dt * kx1 / 2, z + dt * kz1 / 2)
        kx3, kz3 = rhs(x + dt * kx2 / 2, z + dt * kz2 / 2)
        kx4, kz4 = rhs(x + dt * kx3, z + dt * kz3)
        x += dt * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
        z += dt * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    targets = np.array([row['y'] for row in rows], dtype=float)

    def residual(p):
        return np.array([_forecast(row, p) for row in rows]) - targets

    _params = least_squares(residual, [1.8, 1.0, 1.0, 0.5], bounds=([0, 0, 0, 0.02], [np.inf, np.inf, np.inf, np.inf]), x_scale='jac', gtol=1e-12).x
    return _params


def predict(row):
    return _forecast(row, _params)
