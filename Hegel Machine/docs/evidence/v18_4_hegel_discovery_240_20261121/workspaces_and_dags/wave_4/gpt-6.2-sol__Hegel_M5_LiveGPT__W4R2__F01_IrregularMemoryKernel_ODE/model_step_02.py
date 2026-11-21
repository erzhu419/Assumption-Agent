import math
import numpy as np
from scipy.optimize import least_squares


def _prepare(row):
    mem = row.get('mem_conv')
    history = None
    if mem is None:
        prefix = row['raw_prefix']
        t = np.array([v['t'] for v in prefix], dtype=float)
        x = np.array([v['x'] for v in prefix], dtype=float)
        history = (t[-1] - t, np.diff(t), x)
    return float(row['x_now']), float(row['u_now']), float(row['H']), mem, history


def _forecast(p, data):
    a, b, c, tau = map(float, p)
    x, u, H, mem, history = data
    if mem is None:
        lag, dt, x_hist = history
        w = np.exp(-lag / tau)
        wx = w * x_hist
        z = float(0.52 * (wx[0] + np.dot(dt, wx[:-1] + wx[1:]) / (2.0 * tau)))
    else:
        z = float(mem)
    n = max(1, math.ceil(H / 0.004))
    h = H / n
    drive = c * u

    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z)/tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + h*k1/2, z + h*l1/2)
        k3, l3 = rhs(x + h*k2/2, z + h*l2/2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += h*(k1 + 2*k2 + 2*k3 + k4)/6
        z += h*(l1 + 2*l2 + 2*l3 + l4)/6
    return float(x)


def fit(train_rows):
    global _params
    rows = [_prepare(row) for row in train_rows]
    y = np.array([row['y'] for row in train_rows], dtype=float)

    def residual(p):
        return np.array([_forecast(p, row) for row in rows]) - y

    _params = least_squares(residual, [2.0, 1.3, 1.2, 0.6],
                            bounds=([0.0, 0.0, 0.0, 0.02], np.inf),
                            x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return _params


def predict(row):
    return _forecast(_params, _prepare(row))
