import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    memory = row.get('mem_conv')
    if memory is None:
        prefix = row['raw_prefix']
        t = np.asarray([p['t'] for p in prefix], dtype=float)
        x = np.asarray([p['x'] for p in prefix], dtype=float)
        memory = (t[-1] - t, np.diff(t), x)
    return float(row['x_now']), float(row['u_now']), float(row['H']), memory


def _rk4_matrix(A, h):
    B = h * A
    I = np.eye(3)
    return I + B @ (I + B @ (I / 2 + B @ (I / 6 + B / 24)))


def _forecast(rows, p):
    a, b, c, tau = p
    A = np.array([[-a, b, c], [0.52 / tau, -1 / tau, 0.0], [0.0, 0.0, 0.0]])
    step = _rk4_matrix(A, 0.004)
    for x, u, horizon, memory in rows:
        if isinstance(memory, tuple):
            lag, dt, hist_x = memory
            w = np.exp(-lag / tau) * hist_x
            z = 0.52 * (w[0] + np.dot(dt, w[:-1] + w[1:]) / (2 * tau))
        else:
            z = float(memory)
        n = math.floor(horizon / 0.004)
        state = np.linalg.matrix_power(step, n) @ np.array([x, z, u])
        rem = horizon - n * 0.004
        if rem > 0:
            state = _rk4_matrix(A, rem) @ state
        yield float(state[0])


def fit(train_rows):
    global _params
    train_rows = list(train_rows)
    rows = [_prepare(row) for row in train_rows]
    y = np.asarray([row['y'] for row in train_rows], dtype=float)

    def residual(p):
        return np.fromiter(_forecast(rows, p), dtype=float, count=len(rows)) - y

    fits = [least_squares(residual, [2.0, 1.0, 1.5, tau],
                          bounds=([0.0, 0.0, 0.0, 0.01], [np.inf] * 4),
                          x_scale='jac', max_nfev=500)
            for tau in (0.2, 0.8)]
    _params = min(fits, key=lambda result: result.cost).x
    return _params


def predict(row) -> float:
    return next(_forecast([_prepare(row)], _params))