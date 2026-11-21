import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    z = row.get('mem_conv')
    intervals = ()
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * prefix[0]['x']
        intervals = tuple((r['t'] - l['t'], l['x'], r['x']) for l, r in zip(prefix, prefix[1:]))
    H = float(row['H'])
    n = max(1, math.ceil(H / 0.004))
    return float(row['x_now']), float(row['u_now']), float(z), intervals, n, H / n

def _forecast(p, data):
    a, b, c, tau = map(float, p)
    x, u, z, intervals, n, h = data
    inv_tau = 1.0 / tau
    for dt, xl, xr in intervals:
        q = dt * inv_tau
        decay = math.exp(-q)
        z = decay * z + 0.26 * q * (decay * xl + xr)
    drive = c * u
    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) * inv_tau
    hh = h / 2.0
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + hh * k1, z + hh * l1)
        k3, l3 = rhs(x + hh * k2, z + hh * l2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x += h / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
        z += h / 6.0 * (l1 + 2.0 * (l2 + l3) + l4)
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = [_prepare(row) for row in rows]
    targets = np.array([row['y'] for row in rows], dtype=float)
    def residual(p):
        return np.array([_forecast(p, d) for d in data]) - targets
    result = least_squares(residual, [1.5, 0.8, 1.0, 0.6], bounds=([0.0, 0.0, 0.0, 0.01], [np.inf] * 4), x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10)
    _params = result.x
    return _params

def predict(row) -> float:
    return _forecast(_params, _prepare(row))
