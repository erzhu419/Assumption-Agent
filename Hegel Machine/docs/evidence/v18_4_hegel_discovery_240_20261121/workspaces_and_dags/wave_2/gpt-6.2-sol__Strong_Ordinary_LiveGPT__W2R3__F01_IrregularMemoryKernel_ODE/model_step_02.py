import math
import numpy as np
from scipy.optimize import least_squares


def _prepare(row):
    z = row.get('mem_conv')
    prefix = row['raw_prefix'] if z is None else ()
    intervals = [(right['t'] - left['t'], left['x'], right['x'])
                 for left, right in zip(prefix, prefix[1:])]
    if z is None:
        z = 0.52 * prefix[0]['x']
    h = float(row['H'])
    n = max(1, math.ceil(h / 0.004))
    return float(row['x_now']), float(z), float(row['u_now']), intervals, h / n, n


def _forecast(params, data):
    a, b, c, tau = map(float, params)
    x, z, u, intervals, dt, n = data
    for width, x_left, x_right in intervals:
        q = width / tau
        decay = math.exp(-q)
        z = decay * z + 0.26 * q * (decay * x_left + x_right)

    drive = c * u
    def rates(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    half = dt / 2.0
    for _ in range(n):
        k1, l1 = rates(x, z)
        k2, l2 = rates(x + half*k1, z + half*l1)
        k3, l3 = rates(x + half*k2, z + half*l2)
        k4, l4 = rates(x + dt*k3, z + dt*l3)
        x += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        z += dt * (l1 + 2*l2 + 2*l3 + l4) / 6.0
    return x


def fit(train_rows):
    global _params
    rows = [(_prepare(row), float(row['y'])) for row in train_rows]
    def residuals(params):
        return np.array([_forecast(params, data) - y for data, y in rows])
    _params = least_squares(residuals, [1.5, 1.0, 1.3, 0.5],
                            bounds=([0.0, 0.0, 0.0, 0.01], np.inf),
                            gtol=1e-11).x
    return _params


def predict(row) -> float:
    return float(_forecast(_params, _prepare(row)))