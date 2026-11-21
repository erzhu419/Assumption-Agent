import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _forecast(row, params):
    a, b, c, tau = params
    x = float(row['x_now'])
    z = row.get('mem_conv')
    if z is None:
        prefix = sorted(row.get('raw_prefix') or (), key=lambda point: point['t'])
        z = 0.52 * (float(prefix[0]['x']) if prefix else x)
        for left, right in zip(prefix, prefix[1:]):
            q = (float(right['t']) - float(left['t'])) / tau
            decay = math.exp(-q)
            z = decay * z + 0.26 * q * (decay * float(left['x']) + float(right['x']))
    z = float(z)
    drive = c * float(row['u_now'])
    horizon = float(row['H'])
    steps = max(1, math.ceil(horizon / 0.004))
    h = horizon / steps

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(steps):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5 * h * kx, z + 0.5 * h * kz)
        mx, mz = rhs(x + 0.5 * h * lx, z + 0.5 * h * lz)
        nx, nz = rhs(x + h * mx, z + h * mz)
        x += h * (kx + 2 * lx + 2 * mx + nx) / 6
        z += h * (kz + 2 * lz + 2 * mz + nz) / 6
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    targets = np.array([float(row['y']) for row in rows])

    def residuals(params):
        return np.array([_forecast(row, params) for row in rows]) - targets

    result = least_squares(residuals, [1.8, 1.2, 1.2, 0.5], bounds=([0.0, 0.0, 0.0, 0.01], np.inf), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params


def predict(row) -> float:
    return _forecast(row, _params)
