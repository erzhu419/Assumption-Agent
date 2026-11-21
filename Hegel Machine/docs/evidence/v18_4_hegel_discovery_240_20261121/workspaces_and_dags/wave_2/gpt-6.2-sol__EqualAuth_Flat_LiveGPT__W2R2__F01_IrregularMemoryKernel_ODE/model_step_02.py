import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_I = np.eye(3)


def _initial_memory(row, tau):
    memory = row.get('mem_conv')
    if memory is not None:
        return float(memory)
    prefix = row.get('raw_prefix')
    if not prefix:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(prefix[0]['x'])
    for left, right in zip(prefix, prefix[1:]):
        d = (float(right['t']) - float(left['t'])) / tau
        if d == 0.0:
            continue
        w = -math.expm1(-d)
        x_left = float(left['x'])
        dx = float(right['x']) - x_left
        # Integrate the exponential kernel exactly for linear x on this interval.
        z = math.exp(-d) * z + 0.52 * (w * x_left + (1.0 - w / d) * dx)
    return z


def _forecast(row, params):
    a, b, c, tau = params
    x = float(row['x_now'])
    horizon = float(row['H'])
    if horizon == 0.0:
        return x
    z = _initial_memory(row, tau)
    A = np.array([[-a, b, c * float(row['u_now'])], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    n = max(1, math.ceil(horizon / 0.004))
    B = (horizon / n) * A
    # One classical RK4 step for the augmented state [x, z, 1].
    step = _I + B @ (_I + B @ (_I / 2.0 + B @ (_I / 6.0 + B / 24.0)))
    return float((np.linalg.matrix_power(step, n) @ np.array([x, z, 1.0]))[0])


def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])

    def residual(params):
        return np.array([_forecast(row, params) for row in rows]) - y

    result = least_squares(residual, [1.5, 1.0, 1.2, 0.6], bounds=([0.0, 0.0, 0.0, 0.02], [8.0, 8.0, 8.0, 5.0]), x_scale='jac')
    _params = result.x
    return _params


def predict(row):
    return _forecast(row, _params)
