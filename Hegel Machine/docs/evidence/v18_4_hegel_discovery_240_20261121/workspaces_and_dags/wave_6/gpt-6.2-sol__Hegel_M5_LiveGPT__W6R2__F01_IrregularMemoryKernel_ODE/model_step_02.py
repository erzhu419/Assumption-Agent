import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    z = row.get('mem_conv')
    history = None
    if z is None:
        prefix = row['raw_prefix']
        t = np.asarray([p['t'] for p in prefix], dtype=float)
        x = np.asarray([p['x'] for p in prefix], dtype=float)
        history = (t[-1] - t, np.diff(t), x)
    return float(row['x_now']), float(row['u_now']), float(row['H']), z, history


def _forecast(data, params):
    x, u, horizon, z, history = data
    a, b, c, tau = map(float, params)
    if z is None:
        age, widths, past_x = history
        weighted = np.exp(-age / tau) * past_x
        integral = 0.5 * np.dot(widths, weighted[:-1] + weighted[1:])
        z = 0.52 * (weighted[0] + integral / tau)
    z = float(z)
    drive = c * u
    inv_tau = 1.0 / tau
    n = math.ceil(horizon / 0.004)
    for j in range(n):
        h = min(0.004, horizon - 0.004 * j)
        k1 = -a*x + b*z + drive
        l1 = (0.52*x - z) * inv_tau
        x2, z2 = x + 0.5*h*k1, z + 0.5*h*l1
        k2 = -a*x2 + b*z2 + drive
        l2 = (0.52*x2 - z2) * inv_tau
        x3, z3 = x + 0.5*h*k2, z + 0.5*h*l2
        k3 = -a*x3 + b*z3 + drive
        l3 = (0.52*x3 - z3) * inv_tau
        x4, z4 = x + h*k3, z + h*l3
        k4 = -a*x4 + b*z4 + drive
        l4 = (0.52*x4 - z4) * inv_tau
        x += h*(k1 + 2*k2 + 2*k3 + k4)/6
        z += h*(l1 + 2*l2 + 2*l3 + l4)/6
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = [_prepare(row) for row in rows]
    y = np.asarray([row['y'] for row in rows], dtype=float)

    def residual(params):
        return np.asarray([_forecast(d, params) for d in data]) - y

    best = None
    for tau0 in (0.2, 0.07, 0.8):
        result = least_squares(residual, [1.6, 0.8, 1.5, tau0],
                               bounds=([0, 0, 0, 0.01], [20, 20, 20, 10]),
                               x_scale='jac', max_nfev=400,
                               ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if math.sqrt(2 * best.cost / len(y)) < 1e-7:
            break
    _params = best.x
    return _params


def predict(row) -> float:
    return _forecast(_prepare(row), _params)
