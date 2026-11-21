import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    memory = row.get('mem_conv')
    history = None
    if memory is None:
        prefix = row['raw_prefix']
        values = [float(p['x']) for p in prefix]
        gaps = [float(q['t']) - float(p['t']) for p, q in zip(prefix, prefix[1:])]
        history = (values[0], list(zip(gaps, values, values[1:])))
    return float(row['x_now']), float(row['u_now']), float(row['H']), memory, history


def _rk4_matrix(A, h):
    B = h * A
    I = np.eye(3)
    return I + B @ (I + B @ (I / 2 + B @ (I / 6 + B / 24)))


def _forecast(data, params):
    x, u, H, z, history = data
    a, b, c, tau = params
    if z is None:
        first, intervals = history
        z = 0.52 * first
        # Exact kernel integral when x is linear between consecutive samples.
        for dt, left, right in intervals:
            q = dt / tau
            gain = -math.expm1(-q)
            z = math.exp(-q) * z + 0.52 * (gain * left + (1 - gain / q) * (right - left))
    # Augment the state with a constant 1 to include the held input in RK4.
    A = np.array([[-a, b, c * u], [0.52 / tau, -1 / tau, 0.0], [0.0, 0.0, 0.0]])
    dt = 0.004
    n = int(H / dt)
    state = np.linalg.matrix_power(_rk4_matrix(A, dt), n) @ np.array([x, float(z), 1.0])
    remainder = H - n * dt
    if remainder > 0:
        state = _rk4_matrix(A, remainder) @ state
    return float(state[0])


def fit(train_rows):
    global _params
    data = [_prepare(row) for row in train_rows]
    targets = np.array([float(row['y']) for row in train_rows])

    def residual(params):
        return np.array([_forecast(d, params) for d in data]) - targets

    result = least_squares(residual, [1.5, 1.0, 1.4, 0.5], bounds=([0, 0, 0, 0.02], [20, 20, 20, 10]), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params


def predict(row) -> float:
    return _forecast(_prepare(row), _params)
