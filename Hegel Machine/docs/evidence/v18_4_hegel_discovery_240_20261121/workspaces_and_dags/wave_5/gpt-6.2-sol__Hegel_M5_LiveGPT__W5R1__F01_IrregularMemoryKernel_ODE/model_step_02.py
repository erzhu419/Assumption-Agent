import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _pack(row):
    z = row['mem_conv']
    if z is not None:
        hist = None
    else:
        pref = row['raw_prefix']
        t = np.array([p['t'] for p in pref], dtype=float)
        x = np.array([p['x'] for p in pref], dtype=float)
        hist = (t[-1] - t, np.diff(t), x)
    return float(row['x_now']), float(row['u_now']), float(row['H']), z, hist


def _forecast(p, data):
    a, b, c, tau = p
    x, u, H, z, hist = data
    if z is None:
        age, dt, xp = hist
        w = xp * np.exp(-age / tau)
        z = 0.52 * (w[0] + np.dot(dt, w[:-1] + w[1:]) / (2.0 * tau))
    if H == 0.0:
        return x
    n = math.ceil(H / 0.004)
    h = H / n
    drive = c * u

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + h * k1 / 2, z + h * l1 / 2)
        k3, l3 = rhs(x + h * k2 / 2, z + h * l2 / 2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        z += h * (l1 + 2 * l2 + 2 * l3 + l4) / 6
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = [_pack(r) for r in rows]
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        return np.array([_forecast(p, d) for d in data]) - y

    fits = [least_squares(residual, [1.6, 0.9, 1.1, tau],
                         bounds=([0.0, 0.0, 0.0, 0.01], [np.inf] * 4),
                         x_scale='jac', max_nfev=250)
            for tau in (0.12, 0.45, 1.2)]
    _params = min(fits, key=lambda result: result.cost).x
    return _params


def predict(row) -> float:
    return _forecast(_params, _pack(row))
