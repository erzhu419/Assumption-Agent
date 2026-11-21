import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    x = float(row['x_now'])
    z = row.get('mem_conv')
    prefix = row.get('raw_prefix')
    if z is not None:
        z, intervals = float(z), ()
    elif prefix:
        prefix = sorted(prefix, key=lambda point: point['t'])
        z = 0.52 * prefix[0]['x']
        intervals = tuple((right['t'] - left['t'], left['x'], right['x'] - left['x'])
                          for left, right in zip(prefix, prefix[1:])
                          if right['t'] > left['t'])
    else:
        z, intervals = 0.52 * x, ()
    return x, float(row['u_now']), float(row['H']), z, intervals


def _forecast(prepared, params):
    a, b, c, tau = map(float, params)
    x, u, horizon, z, intervals = prepared
    # Exact exponential convolution for linear interpolation on each interval.
    for dt, x_left, dx in intervals:
        q = dt / tau
        weight = -math.expm1(-q)
        ramp = (q * (0.5 - q / 6.0 + q * q / 24.0)
                if q < 1e-4 else 1.0 - weight / q)
        z = math.exp(-q) * z + 0.52 * (weight * x_left + ramp * dx)
    if horizon == 0.0:
        return x
    n = math.ceil(horizon / min(0.004, 0.25 * tau))
    h = horizon / n
    drive = c * u
    inv_tau = 1.0 / tau
    for _ in range(n):
        k1 = -a * x + b * z + drive
        l1 = (0.52 * x - z) * inv_tau
        k2 = k1 + 0.5 * h * (-a * k1 + b * l1)
        l2 = l1 + 0.5 * h * (0.52 * k1 - l1) * inv_tau
        k3 = k1 + 0.5 * h * (-a * k2 + b * l2)
        l3 = l1 + 0.5 * h * (0.52 * k2 - l2) * inv_tau
        k4 = k1 + h * (-a * k3 + b * l3)
        l4 = l1 + h * (0.52 * k3 - l3) * inv_tau
        x += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z += h / 6.0 * (l1 + 2.0 * l2 + 2.0 * l3 + l4)
    return x


def fit(train_rows):
    global _params
    rows = [_prepare(row) for row in train_rows]
    targets = np.asarray([row['y'] for row in train_rows], dtype=float)

    def residual(params):
        return np.asarray([_forecast(row, params) for row in rows]) - targets

    result = least_squares(residual, [2.0, 1.0, 1.0, 0.2],
                           bounds=([0.0, 0.0, 0.0, 0.008], [np.inf] * 4),
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    _params = result.x


def predict(row) -> float:
    return float(_forecast(_prepare(row), _params))