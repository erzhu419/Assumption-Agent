import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    mem = row.get('mem_conv')
    history = None
    if mem is None:
        prefix = row['raw_prefix']
        end = float(prefix[-1]['t'])
        age = np.array([end - float(p['t']) for p in prefix])
        values = np.array([float(p['x']) for p in prefix])
        history = (age, values, age[:-1] - age[1:])
    return float(row['x_now']), float(row['u_now']), float(row['H']), mem, history


def _forecast(params, prepared):
    a, b, c, tau = map(float, params)
    x, u, horizon, mem, history = prepared
    if mem is None:
        age, values, intervals = history
        weighted = np.exp(-age / tau) * values
        z = 0.52 * (weighted[0] + np.dot(intervals, weighted[:-1] + weighted[1:]) / (2.0 * tau))
    else:
        z = float(mem)
    drive = c * u

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    n = max(1, math.ceil(horizon / 0.004))
    dt = horizon / n
    half = 0.5 * dt
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + half * k1, z + half * l1)
        k3, l3 = rhs(x + half * k2, z + half * l2)
        k4, l4 = rhs(x + dt * k3, z + dt * l3)
        x += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z += dt / 6.0 * (l1 + 2.0 * l2 + 2.0 * l3 + l4)
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    prepared = [_prepare(row) for row in rows]
    observed = np.array([float(row['y']) for row in rows])

    def residual(params):
        return np.array([_forecast(params, row) for row in prepared]) - observed

    result = least_squares(residual, [1.5, 1.0, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.01], np.inf), x_scale='jac', gtol=1e-11)
    _params = result.x
    return _params


def predict(row):
    return _forecast(_params, _prepare(row))
