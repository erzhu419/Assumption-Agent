import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _memory(row, tau):
    measured = row.get('mem_conv')
    if measured is not None:
        return float(measured)
    prefix = row.get('raw_prefix')
    if not prefix:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(prefix[0]['x'])
    for left, right in zip(prefix, prefix[1:]):
        ratio = (float(right['t']) - float(left['t'])) / tau
        decay = math.exp(-ratio)
        z = decay * z + 0.26 * ratio * (decay * float(left['x']) + float(right['x']))
    return z


def _forecast(row, params):
    a, b, c, tau = map(float, params)
    x = float(row['x_now'])
    z = _memory(row, tau)
    horizon = float(row['H'])
    n = max(1, math.ceil(horizon / min(0.004, 0.25 * tau)))
    dt = horizon / n
    drive = c * float(row['u_now'])

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * dt * k1, z + 0.5 * dt * l1)
        k3, l3 = rhs(x + 0.5 * dt * k2, z + 0.5 * dt * l2)
        k4, l4 = rhs(x + dt * k3, z + dt * l3)
        x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        z += dt * (l1 + 2 * l2 + 2 * l3 + l4) / 6
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])

    def residuals(params):
        return np.array([_forecast(row, params) for row in rows]) - y

    result = least_squares(residuals, [1.6, 1.1, 1.2, 0.6], bounds=([0, 0, 0, 0.02], [20, 20, 20, 10]), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params


def predict(row) -> float:
    return _forecast(row, _params)
