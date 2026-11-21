import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    mem = row.get('mem_conv')
    if mem is not None:
        history = float(mem)
    else:
        prefix = row['raw_prefix']
        t = np.array([p['t'] for p in prefix], dtype=float)
        x = np.array([p['x'] for p in prefix], dtype=float)
        history = (t[-1] - t, np.diff(t), x)
    return float(row['x_now']), float(row['u_now']), float(row['H']), history


def _forecast(params, prepared):
    a, b, c, tau = map(float, params)
    x, u, H, history = prepared
    if isinstance(history, tuple):
        age, widths, prefix_x = history
        kernel = np.exp(-age / tau)
        weighted_x = prefix_x * kernel
        z = 0.52 * (weighted_x[0] + np.dot(widths, weighted_x[:-1] + weighted_x[1:]) / (2.0 * tau))
        z = float(z)
    else:
        z = history
    n = max(1, math.ceil(H / 0.004))
    h = H / n
    hh = 0.5 * h
    drive = c * u
    for _ in range(n):
        k1 = -a*x + b*z + drive
        l1 = (0.52*x - z) / tau
        xx, zz = x + hh*k1, z + hh*l1
        k2 = -a*xx + b*zz + drive
        l2 = (0.52*xx - zz) / tau
        xx, zz = x + hh*k2, z + hh*l2
        k3 = -a*xx + b*zz + drive
        l3 = (0.52*xx - zz) / tau
        xx, zz = x + h*k3, z + h*l3
        k4 = -a*xx + b*zz + drive
        l4 = (0.52*xx - zz) / tau
        x += (h/6.0)*(k1 + 2.0*(k2 + k3) + k4)
        z += (h/6.0)*(l1 + 2.0*(l2 + l3) + l4)
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    prepared = [_prepare(row) for row in rows]
    y = np.array([row['y'] for row in rows], dtype=float)

    def residual(params):
        return np.array([_forecast(params, row) for row in prepared]) - y

    _params = least_squares(residual, [1.4, 0.7, 1.4, 0.5], bounds=([0.0, 0.0, 0.0, 0.01], np.inf), x_scale='jac').x
    return _params


def predict(row) -> float:
    return float(_forecast(_params, _prepare(row)))