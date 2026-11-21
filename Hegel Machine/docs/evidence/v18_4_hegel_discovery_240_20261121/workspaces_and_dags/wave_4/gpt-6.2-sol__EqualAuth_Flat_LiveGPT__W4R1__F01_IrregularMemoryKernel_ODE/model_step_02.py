import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _forecast(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    z = row.get('mem_conv')
    if z is None:
        prefix = sorted(row['raw_prefix'], key=lambda sample: sample['t'])
        z = 0.52 * prefix[0]['x']
        for left, right in zip(prefix, prefix[1:]):
            w = (right['t'] - left['t']) / tau
            if w == 0.0:
                continue
            mass = -math.expm1(-w)
            # Exact exponential convolution for linear interpolation on this interval.
            z = math.exp(-w) * z + 0.52 * (left['x'] * mass + (right['x'] - left['x']) * (1.0 - mass / w))
    z = float(z)
    if H == 0.0:
        return x
    n = math.ceil(H / 0.004)
    h = H / n
    drive = c * u

    def rhs(X, Z):
        return -a * X + b * Z + drive, (0.52 * X - Z) / tau

    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5 * h * k1x, z + 0.5 * h * k1z)
        k3x, k3z = rhs(x + 0.5 * h * k2x, z + 0.5 * h * k2z)
        k4x, k4z = rhs(x + h * k3x, z + h * k3z)
        x += h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        z += h * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([row['y'] for row in rows], dtype=float)
    result = least_squares(lambda p: np.array([_forecast(row, p) for row in rows]) - y,
                           x0=[1.6, 1.0, 1.5, 0.5],
                           bounds=([0.0, 0.0, 0.0, 0.01], [np.inf] * 4))
    _params = result.x
    return _params


def predict(row) -> float:
    return _forecast(row, _params)
