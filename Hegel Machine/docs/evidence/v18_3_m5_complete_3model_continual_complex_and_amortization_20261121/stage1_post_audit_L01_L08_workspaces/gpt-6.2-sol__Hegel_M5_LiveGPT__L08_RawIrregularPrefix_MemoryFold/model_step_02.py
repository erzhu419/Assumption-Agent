import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        t = np.array([p['t'] - prefix[-1]['t'] for p in prefix], dtype=float)
        x = np.array([p['x'] for p in prefix], dtype=float)
        history = (t, x, np.diff(t))
    else:
        history = None
    return float(row['x_now']), float(row['u_now']), float(row['H']), z, history


def _rk4_matrix(A, h):
    M = h * A
    I = np.eye(3)
    return I + M @ (I + M @ (I / 2 + M @ (I / 6 + M / 24)))


def _forecast(data, params):
    a, b, c, tau = params
    x0, u, H, z0, history = data
    if history is not None:
        t, x, dt = history
        w = np.exp(t / tau) * x
        z0 = 0.52 * (math.exp(t[0] / tau) * x[0] + np.dot(dt, w[:-1] + w[1:]) / (2 * tau))
    A = np.array([[-a, b, c * u], [0.52 / tau, -1 / tau, 0.0], [0.0, 0.0, 0.0]])
    h = 0.004
    n = int(H / h)
    state = np.linalg.matrix_power(_rk4_matrix(A, h), n) @ np.array([x0, z0, 1.0])
    remainder = H - n * h
    if remainder > 0:
        state = _rk4_matrix(A, remainder) @ state
    return float(state[0])


def fit(train_rows):
    global _params
    rows = list(train_rows)
    inputs = [_prepare(row) for row in rows]
    y = np.array([row['y'] for row in rows], dtype=float)

    def residual(params):
        return np.array([_forecast(data, params) for data in inputs]) - y

    _params = least_squares(residual, [1.8, 1.3, 1.35, 0.6], bounds=([0, 0, 0, 0.01], np.inf), gtol=1e-12).x
    return _params


def predict(row) -> float:
    return _forecast(_prepare(row), _params)
