import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    z = row.get('mem_conv')
    prefix = None
    if z is None:
        points = sorted(row['raw_prefix'], key=lambda p: p['t'])
        xs = np.array([p['x'] for p in points], dtype=float)
        dt = np.diff(np.array([p['t'] for p in points], dtype=float))
        prefix = (xs, dt)
    return float(row['x_now']), float(row['u_now']), float(row['H']), z, prefix

def _forecast(data, params):
    x, u, H, z, prefix = data
    if H == 0.0:
        return x
    a, b, c, tau = params
    if z is None:
        xs, dt = prefix
        z = 0.52 * xs[0]
        for left, right, width in zip(xs[:-1], xs[1:], dt):
            r = width / tau
            decay = math.exp(-r)
            z = decay * z + 0.26 * r * (decay * left + right)
    n = max(1, math.ceil(H / 0.004))
    # The constant-input RK4 step is a matrix acting on (x, z, u).
    B = (H / n) * np.array([[-a, b, c], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    eye = np.eye(3)
    step = eye + B @ (eye + B @ (eye / 2.0 + B @ (eye / 6.0 + B / 24.0)))
    return float(np.linalg.matrix_power(step, n)[0] @ np.array([x, z, u]))

def fit(train_rows):
    global _params
    data = [_prepare(row) for row in train_rows]
    targets = np.array([row['y'] for row in train_rows], dtype=float)
    def residual(params):
        return np.array([_forecast(row, params) for row in data]) - targets
    result = least_squares(residual, [1.8, 1.3, 1.35, 0.6], bounds=([0.0, 0.0, 0.0, 0.02], np.inf), x_scale='jac')
    _params = result.x
    return _params

def predict(row):
    return _forecast(_prepare(row), _params)
