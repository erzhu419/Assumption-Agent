import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _forecast(row, params):
    a, b, c, tau = map(float, params)
    x, u, horizon = (float(row[k]) for k in ('x_now', 'u_now', 'H'))
    z = row.get('mem_conv')
    if z is None:
        prefix = row.get('raw_prefix')
        z = 0.52 * (prefix[0]['x'] if prefix else x)
        if prefix:
            for left, right in zip(prefix, prefix[1:]):
                q = (right['t'] - left['t']) / tau
                e = math.exp(-q)
                z = e * z + 0.26 * q * (e * left['x'] + right['x'])
    else:
        z = float(z)

    drive = c * u

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    steps = max(1, math.ceil(horizon / 0.004))
    h = horizon / steps
    for _ in range(steps):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * h * k1, z + 0.5 * h * l1)
        k3, l3 = rhs(x + 0.5 * h * k2, z + 0.5 * h * l2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        z += h * (l1 + 2 * l2 + 2 * l3 + l4) / 6
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([row['y'] for row in rows], dtype=float)

    def residuals(params):
        return np.array([_forecast(row, params) for row in rows]) - y

    _params = least_squares(residuals, [1.5, 0.6, 0.8, 0.5], bounds=([0, 0, 0, 0.02], np.inf), gtol=1e-12).x


def predict(row) -> float:
    return _forecast(row, _params)
