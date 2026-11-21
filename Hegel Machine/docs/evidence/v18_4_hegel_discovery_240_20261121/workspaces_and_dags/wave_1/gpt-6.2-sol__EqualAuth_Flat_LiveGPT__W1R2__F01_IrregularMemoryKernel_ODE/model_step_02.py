import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        t_end = float(prefix[-1]['t'])
        t = np.array([float(p['t']) - t_end for p in prefix])
        v = np.array([0.52 * float(p['x']) for p in prefix])
        memory = (t, np.diff(t), v)
    else:
        memory = float(z)
    return float(row['x_now']), float(row['u_now']), float(row['H']), memory

def _forecast(p, data):
    a, b, c, tau = map(float, p)
    x, u, horizon, z = data
    if isinstance(z, tuple):
        t, gaps, v = z
        weighted = np.exp(t / tau) * v
        z = float(weighted[0] + np.dot(gaps, weighted[:-1] + weighted[1:]) / (2.0 * tau))
    n = max(1, math.ceil(horizon / min(0.004, tau / 4.0)))
    h = horizon / n
    drive = c * u

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + h * kx / 2, z + h * kz / 2)
        mx, mz = rhs(x + h * lx / 2, z + h * lz / 2)
        nx, nz = rhs(x + h * mx, z + h * mz)
        x += h * (kx + 2 * lx + 2 * mx + nx) / 6
        z += h * (kz + 2 * lz + 2 * mz + nz) / 6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = [_prepare(row) for row in rows]
    targets = np.array([float(row['y']) for row in rows])

    def residual(p):
        return np.array([_forecast(p, d) for d in data]) - targets

    result = least_squares(residual, [1.0, 0.5, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.02], [np.inf] * 4), x_scale='jac', gtol=1e-12)
    _params = result.x

def predict(row):
    return _forecast(_params, _prepare(row))
