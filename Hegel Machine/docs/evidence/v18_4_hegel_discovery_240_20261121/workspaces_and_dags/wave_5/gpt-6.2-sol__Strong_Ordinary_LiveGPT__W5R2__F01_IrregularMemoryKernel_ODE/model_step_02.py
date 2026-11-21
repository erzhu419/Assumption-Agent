import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _forecast(row, params):
    a, b, c, tau = params
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * prefix[0]['x']
        for left, right in zip(prefix, prefix[1:]):
            q = (right['t'] - left['t']) / tau
            decay = math.exp(-q)
            z = decay * z + 0.26 * q * (decay * left['x'] + right['x'])
    n = max(1, math.ceil(H / 0.004))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + c * u, (0.52 * x - z) / tau

    for _ in range(n):
        kx1, kz1 = rhs(x, z)
        kx2, kz2 = rhs(x + 0.5*h*kx1, z + 0.5*h*kz1)
        kx3, kz3 = rhs(x + 0.5*h*kx2, z + 0.5*h*kz2)
        kx4, kz4 = rhs(x + h*kx3, z + h*kz3)
        x += (h/6) * (kx1 + 2*kx2 + 2*kx3 + kx4)
        z += (h/6) * (kz1 + 2*kz2 + 2*kz3 + kz4)
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([row['y'] for row in rows], dtype=float)

    def residual(params):
        return np.array([_forecast(row, params) for row in rows]) - y

    result = least_squares(residual, [1.5, 1.0, 1.0, 0.4],
                           bounds=([0.0, 0.0, 0.0, 0.01], [20.0, 20.0, 20.0, 20.0]))
    _params = result.x
    return _params


def predict(row) -> float:
    return _forecast(row, _params)
