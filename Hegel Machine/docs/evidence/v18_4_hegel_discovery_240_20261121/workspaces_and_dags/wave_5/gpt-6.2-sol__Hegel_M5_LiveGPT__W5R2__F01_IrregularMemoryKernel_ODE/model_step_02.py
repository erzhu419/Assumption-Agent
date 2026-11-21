import math
import numpy as np
from scipy.optimize import least_squares


def _prepare(row):
    memory = row.get('mem_conv')
    if memory is not None:
        history = float(memory)
    else:
        prefix = row['raw_prefix']
        t = np.array([p['t'] for p in prefix], dtype=float)
        x = np.array([p['x'] for p in prefix], dtype=float)
        history = (t[-1] - t, np.diff(t), x)
    return float(row['x_now']), float(row['u_now']), float(row['H']), history


def _forecast(p, prepared):
    a, b, c, tau = map(float, p)
    x, u, H, history = prepared
    if isinstance(history, tuple):
        age, dt, values = history
        f = np.exp(-age / tau) * values
        z = 0.52 * (f[0] + np.dot(dt, f[:-1] + f[1:]) / (2.0 * tau))
    else:
        z = history
    if H == 0.0:
        return x
    n = max(1, math.ceil(H / 0.004))
    h = H / n
    drive = c * u

    def rhs(q, m):
        return -a * q + b * m + drive, (0.52 * q - m) / tau

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

    best = None
    for tau0 in (0.18, 0.06, 0.6):
        result = least_squares(residual, [2.0, 1.0, 1.5, tau0],
                               bounds=([0., 0., 0., 0.005], [50., 50., 50., np.inf]),
                               x_scale='jac', max_nfev=300,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 0.5 * len(rows) * 1e-12:
            break
    _params = best.x


def predict(row) -> float:
    return _forecast(_params, _prepare(row))