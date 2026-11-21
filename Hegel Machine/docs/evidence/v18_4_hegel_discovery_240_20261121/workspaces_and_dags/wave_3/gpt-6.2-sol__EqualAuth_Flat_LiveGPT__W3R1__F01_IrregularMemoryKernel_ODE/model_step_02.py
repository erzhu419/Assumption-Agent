import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _forecast(row, params):
    a, b, c, tau = map(float, params)
    x, u, H = (float(row[k]) for k in ('x_now', 'u_now', 'H'))
    z = row.get('mem_conv')
    if z is None:
        prefix = row.get('raw_prefix')
        if prefix:
            t = np.array([p['t'] for p in prefix], dtype=float)
            X = np.array([p['x'] for p in prefix], dtype=float)
            weights = np.exp((t - t[-1]) / tau)
            weighted_x = X * weights
            z = 0.52 * X[0] * weights[0] + (0.26 / tau) * np.dot(np.diff(t), weighted_x[:-1] + weighted_x[1:])
        else:
            z = 0.52 * x
    z = float(z)
    n = max(1, math.ceil(H / 0.004))
    h = H / n
    drive = c * u

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5 * h * kx, z + 0.5 * h * kz)
        mx, mz = rhs(x + 0.5 * h * lx, z + 0.5 * h * lz)
        nx, nz = rhs(x + h * mx, z + h * mz)
        x += h * (kx + 2 * lx + 2 * mx + nx) / 6
        z += h * (kz + 2 * lz + 2 * mz + nz) / 6
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(params):
        return np.array([_forecast(r, params) for r in rows]) - y

    _params = least_squares(residual, [1.5, 1.0, 1.0, 0.6], bounds=([0, 0, 0, 0.02], [np.inf, np.inf, np.inf, np.inf]), x_scale='jac').x
    return _params


def predict(row) -> float:
    return _forecast(row, _params)
