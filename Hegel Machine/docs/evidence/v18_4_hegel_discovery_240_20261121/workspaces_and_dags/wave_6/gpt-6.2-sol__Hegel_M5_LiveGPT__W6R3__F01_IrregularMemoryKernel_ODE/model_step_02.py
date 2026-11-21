import math
import numpy as np
from scipy.optimize import least_squares

_STEP = 0.004
_I = np.eye(3)
_params = None


def _prepare(row):
    z = row.get('mem_conv')
    history = None
    if z is None:
        prefix = row['raw_prefix']
        t = np.array([p['t'] for p in prefix], dtype=float)
        x = np.array([p['x'] for p in prefix], dtype=float)
        history = (t[-1] - t, np.diff(t), x)
    H = float(row['H'])
    n = math.floor(H / _STEP)
    return float(row['x_now']), float(row['u_now']), z, history, n, H - n * _STEP


def _rk4_matrix(A, h):
    B = h * A
    return _I + B @ (_I + B @ (_I / 2 + B @ (_I / 6 + B / 24)))


def _evaluate(p, rows):
    a, b, c, tau = p
    A = np.array([[-a, b, c], [0.52 / tau, -1 / tau, 0.0], [0.0, 0.0, 0.0]])
    step = _rk4_matrix(A, _STEP)
    values = []
    for x0, u, z, history, n, remainder in rows:
        if z is None:
            ages, widths, x = history
            w = np.exp(-ages / tau)
            wx = w * x
            z = 0.52 * (wx[0] + np.dot(widths, wx[:-1] + wx[1:]) / (2 * tau))
        state = np.linalg.matrix_power(step, n) @ np.array([x0, z, u])
        if remainder > 0:
            state = _rk4_matrix(A, remainder) @ state
        values.append(state[0])
    return np.array(values)


def fit(train_rows):
    global _params
    rows = [_prepare(row) for row in train_rows]
    y = np.array([row['y'] for row in train_rows], dtype=float)
    result = least_squares(lambda p: _evaluate(p, rows) - y, [1.3, 0.65, 1.1, 0.35], bounds=([0, 0, 0, 0.01], [20, 20, 20, 10]), gtol=1e-12)
    _params = result.x


def predict(row) -> float:
    return float(_evaluate(_params, [_prepare(row)])[0])