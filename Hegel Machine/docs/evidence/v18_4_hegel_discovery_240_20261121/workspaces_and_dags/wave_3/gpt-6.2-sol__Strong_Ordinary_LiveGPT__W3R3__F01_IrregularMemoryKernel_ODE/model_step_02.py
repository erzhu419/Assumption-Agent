import math
import numpy as np
from scipy.optimize import least_squares


def _prepare(row):
    x = float(row['x_now'])
    z = row.get('mem_conv')
    prefix = row.get('raw_prefix')
    if z is not None:
        z = float(z)
    elif prefix:
        t = np.array([p['t'] for p in prefix], dtype=float)
        v = 0.52 * np.array([p['x'] for p in prefix], dtype=float)
        z = (t[-1] - t, np.diff(t), v)
    else:
        z = 0.52 * x
    return x, z, float(row['u_now']), float(row['H'])


def _memory(z, tau):
    if not isinstance(z, tuple):
        return z
    age, dt, v = z
    w = v * np.exp(-age / tau)
    return w[0] + np.dot(dt, w[:-1] + w[1:]) / (2.0 * tau)


def _forecast(data, params):
    a, b, c, tau = params
    n = max(1, math.ceil(max(d[3] for d in data) / 0.004))
    mats = np.zeros((len(data), 3, 3))
    states = np.ones((len(data), 3, 1))
    for i, (x, z, u, H) in enumerate(data):
        mats[i] = np.array([[-a, b, c*u], [0.52/tau, -1.0/tau, 0.0], [0.0, 0.0, 0.0]]) * (H/n)
        states[i, :2, 0] = x, _memory(z, tau)
    # For a constant linear system, this polynomial is exactly one RK4 step.
    eye = np.eye(3)
    step = eye + mats @ (eye + mats @ (0.5*eye + mats @ (eye/6.0 + mats/24.0)))
    return (np.linalg.matrix_power(step, n) @ states)[:, 0, 0]


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = [_prepare(row) for row in rows]
    target = np.array([row['y'] for row in rows], dtype=float)
    result = least_squares(lambda p: _forecast(data, p) - target,
                           [1.6, 1.0, 1.1, 0.2],
                           bounds=([0.0, 0.0, 0.0, 0.01], [np.inf]*4),
                           x_scale='jac', max_nfev=500,
                           ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x


def predict(row) -> float:
    return float(_forecast([_prepare(row)], _params)[0])