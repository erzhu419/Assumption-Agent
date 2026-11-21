import math
import numpy as np
from scipy.optimize import least_squares

_DT = 0.004
_I = np.eye(3)
_params = None


def _prepare(row):
    z = row.get('mem_conv')
    history = None
    if z is None:
        prefix = row['raw_prefix']
        ages = float(prefix[-1]['t']) - np.array([r['t'] for r in prefix], dtype=float)
        xs = np.array([r['x'] for r in prefix], dtype=float)
        history = (ages, xs, ages[:-1] - ages[1:])
    h = float(row['H'])
    n = math.floor(h / _DT)
    return float(row['x_now']), float(row['u_now']), z, history, n, h - n * _DT


def _matrix(p):
    a, b, c, tau = p
    return np.array([[-a, b, c], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])


def _rk4_step(matrix, dt):
    q = dt * matrix
    return _I + q @ (_I + q @ (_I / 2.0 + q @ (_I / 6.0 + q / 24.0)))


def _readout(row, tau, matrix, full_step):
    x, u, z, history, n, rem = row
    if z is None:
        ages, xs, widths = history
        weighted = np.exp(-ages / tau) * xs
        z = 0.52 * (weighted[0] + np.dot(widths, weighted[:-1] + weighted[1:]) / (2.0 * tau))
    # Augment the state with the constant drive, so identical RK4 steps can be composed by matrix powers.
    state = np.linalg.matrix_power(full_step, n) @ np.array([x, z, u])
    if rem > 0.0:
        state = _rk4_step(matrix, rem) @ state
    return float(state[0])


def fit(train_rows):
    global _params
    rows = [_prepare(row) for row in train_rows]
    y = np.array([row['y'] for row in train_rows], dtype=float)

    def residuals(p):
        matrix = _matrix(p)
        step = _rk4_step(matrix, _DT)
        return np.array([_readout(row, p[3], matrix, step) for row in rows]) - y

    bounds = ([0.0, 0.0, 0.0, 0.004], [np.inf, np.inf, np.inf, np.inf])
    fits = [least_squares(residuals, [1.4, 0.8, 0.9, tau], bounds=bounds, x_scale='jac', max_nfev=350) for tau in (0.06, 0.18, 0.6)]
    _params = min(fits, key=lambda result: result.cost).x
    return _params


def predict(row) -> float:
    matrix = _matrix(_params)
    return _readout(_prepare(row), _params[3], matrix, _rk4_step(matrix, _DT))