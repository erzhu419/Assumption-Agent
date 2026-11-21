import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _initial_memory(row, tau):
    if row['mem_conv'] is not None:
        return float(row['mem_conv'])
    prefix = row['raw_prefix']
    z = 0.52 * float(prefix[0]['x'])
    for left, right in zip(prefix, prefix[1:]):
        h = float(right['t']) - float(left['t'])
        decay = math.exp(-h / tau)
        z = decay * z + (0.26 * h / tau) * (decay * float(left['x']) + float(right['x']))
    return z


def _rk4_map(matrix, h):
    # For a linear system, one RK4 step is this fourth-degree matrix polynomial.
    q = h * matrix
    eye = np.eye(3)
    return eye + q @ (eye + q @ (eye / 2.0 + q @ (eye / 6.0 + q / 24.0)))


def _forecast(row, params):
    a, b, vmax, rho, tau = params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    matrix = np.array([[-a, b, drive], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    state = np.array([float(row['x_now']), _initial_memory(row, tau), 1.0])
    dt = 0.004
    n = int(H / dt)
    if n:
        state = np.linalg.matrix_power(_rk4_map(matrix, dt), n) @ state
    remaining = H - n * dt
    if remaining > 0.0:
        state = _rk4_map(matrix, remaining) @ state
    return float(state[0])


def fit(train_rows):
    global _params
    rows = list(train_rows)
    targets = np.array([float(row['y']) for row in rows])

    def residual(params):
        return np.array([_forecast(row, params) for row in rows]) - targets

    result = least_squares(residual, [0.8, 1.2, 6.0, 0.70, 0.3], bounds=([0.0, 0.0, 0.0, 0.0, 0.005], [np.inf, np.inf, np.inf, 1.0, np.inf]), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params


def predict(row) -> float:
    return _forecast(row, _params)
