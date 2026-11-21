import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, math.ceil(H / 0.004))
    z = row['mem_conv']
    intervals = ()
    if z is None:
        prefix = row.get('raw_prefix')
        if prefix:
            z = 0.52 * float(prefix[0]['x'])
            intervals = tuple((float(q['t']) - float(p['t']), float(p['x']), float(q['x'])) for p, q in zip(prefix, prefix[1:]))
        else:
            z = 0.52 * x
    return x, u, n, H / n, float(z), intervals


def _forecast(p, data):
    a, b, c, tau = map(float, p)
    x, u, n, h, z, intervals = data
    for dt, left, right in intervals:
        q = dt / tau
        decay = math.exp(-q)
        z = decay * z + 0.26 * q * (decay * left + right)
    drive = c * u
    inv_tau = 1.0 / tau
    half = 0.5 * h
    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) * inv_tau
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + half * k1, z + half * l1)
        k3, l3 = rhs(x + half * k2, z + half * l2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z += (h / 6.0) * (l1 + 2 * l2 + 2 * l3 + l4)
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = [_prepare(row) for row in rows]
    y = np.array([float(row['y']) for row in rows])
    def residual(p):
        return np.array([_forecast(p, d) for d in data]) - y
    result = least_squares(residual, [1.5, 1.0, 1.5, 0.5], bounds=([0.0, 0.0, 0.0, 0.01], np.inf), gtol=1e-12)
    _params = result.x
    return _params


def predict(row) -> float:
    return float(_forecast(_params, _prepare(row)))
