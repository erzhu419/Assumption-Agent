import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_I = np.eye(3)

def _prepare(row):
    z = row.get('mem_conv')
    history = None
    if z is None:
        prefix = row['raw_prefix']
        end = float(prefix[-1]['t'])
        lag = np.array([end - float(p['t']) for p in prefix])
        x = np.array([float(p['x']) for p in prefix])
        history = (lag, x, lag[:-1] - lag[1:])
    h = float(row['H'])
    n = math.floor(h / 0.004)
    return float(row['x_now']), float(row['u_now']), z, history, n, h - n * 0.004

def _rk4_matrix(A, dt):
    D = dt * A
    return _I + D @ (_I + (D / 2) @ (_I + (D / 3) @ (_I + D / 4)))

def _forecast(item, p):
    a, b, c, tau = p
    x0, u, z0, history, n, remainder = item
    if z0 is None:
        lag, x, widths = history
        weights = np.exp(-lag / tau)
        wx = weights * x
        z0 = 0.52 * (wx[0] + np.dot(widths, wx[:-1] + wx[1:]) / (2 * tau))
    A = np.array([[-a, b, c * u], [0.52 / tau, -1 / tau, 0.0], [0.0, 0.0, 0.0]])
    state = np.linalg.matrix_power(_rk4_matrix(A, 0.004), n) @ np.array([x0, z0, 1.0])
    if remainder > 0:
        state = _rk4_matrix(A, remainder) @ state
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    inputs = [_prepare(row) for row in rows]
    target = np.array([float(row['y']) for row in rows])

    def residual(p):
        return np.array([_forecast(item, p) for item in inputs]) - target

    best = None
    for tau in (0.12, 0.4, 1.0):
        result = least_squares(residual, [1.6, 0.6, 1.0, tau], bounds=([0, 0, 0, 0.015], [20, 20, 20, 10]), x_scale='jac', max_nfev=350)
        if best is None or result.cost < best.cost:
            best = result
    _PARAMS = best.x

def predict(row) -> float:
    return _forecast(_prepare(row), _PARAMS)
