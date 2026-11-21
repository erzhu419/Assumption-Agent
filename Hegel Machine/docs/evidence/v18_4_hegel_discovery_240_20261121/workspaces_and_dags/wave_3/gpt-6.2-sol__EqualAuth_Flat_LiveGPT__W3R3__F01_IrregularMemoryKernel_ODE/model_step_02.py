import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _forecast(row, params):
    a, b, c, tau = params
    x = float(row['x_now'])
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * float(prefix[0]['x'])
        for left, right in zip(prefix, prefix[1:]):
            q = (float(right['t']) - float(left['t'])) / tau
            decay = math.exp(-q)
            z = decay * z + 0.26 * q * (decay * float(left['x']) + float(right['x']))
    else:
        z = float(z)

    horizon = float(row['H'])
    n = max(1, math.ceil(horizon / 0.004))
    h = horizon / n
    drive = c * float(row['u_now'])

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5*h*kx, z + 0.5*h*kz)
        mx, mz = rhs(x + 0.5*h*lx, z + 0.5*h*lz)
        nx, nz = rhs(x + h*mx, z + h*mz)
        x += (h/6) * (kx + 2*lx + 2*mx + nx)
        z += (h/6) * (kz + 2*lz + 2*mz + nz)
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])

    def residual(params):
        return np.array([_forecast(row, params) for row in rows]) - y

    _params = least_squares(residual, [2.0, 1.0, 1.5, 0.5], bounds=([0, 0, 0, 0.01], [np.inf]*4), x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10).x


def predict(row) -> float:
    return _forecast(row, _params)
