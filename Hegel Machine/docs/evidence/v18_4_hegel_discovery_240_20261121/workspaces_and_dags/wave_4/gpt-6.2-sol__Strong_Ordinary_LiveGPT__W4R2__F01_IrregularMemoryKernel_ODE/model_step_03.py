import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _inputs(row):
    x = float(row['x_now'])
    prefix = sorted(row.get('raw_prefix') or [], key=lambda point: point['t'])
    history = None
    if prefix:
        samples = np.array([(p['t'], p['x']) for p in prefix], dtype=float)
        history = (samples[0, 1], np.diff(samples[:, 0]), samples[:-1, 1], np.diff(samples[:, 1]))
    return x, float(row['u_now']), float(row['H']), row.get('mem_conv'), history

def _forecast(inputs, params):
    x, u, horizon, memory, history = inputs
    a, b, c, tau = map(float, params)
    if memory is not None:
        z = float(memory)
    elif history is not None:
        first, intervals, left, changes = history
        z = 0.52 * first
        for dt, x_left, dx in zip(intervals, left, changes):
            q = dt / tau
            if q <= 0.0:
                continue
            weight = -math.expm1(-q)
            z = math.exp(-q) * z + 0.52 * (weight * x_left + (1.0 - weight / q) * dx)
    else:
        z = 0.52 * x

    drive = c * u
    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    steps = max(1, math.ceil(horizon / 0.004))
    h = horizon / steps
    for _ in range(steps):
        kx1, kz1 = rhs(x, z)
        kx2, kz2 = rhs(x + 0.5*h*kx1, z + 0.5*h*kz1)
        kx3, kz3 = rhs(x + 0.5*h*kx2, z + 0.5*h*kz2)
        kx4, kz4 = rhs(x + h*kx3, z + h*kz3)
        x += h*(kx1 + 2*kx2 + 2*kx3 + kx4)/6.0
        z += h*(kz1 + 2*kz2 + 2*kz3 + kz4)/6.0
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    inputs = [_inputs(row) for row in rows]
    targets = np.array([row['y'] for row in rows], dtype=float)
    def residuals(params):
        return np.array([_forecast(row, params) for row in inputs]) - targets
    result = least_squares(residuals, [2.0, 1.0, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.02], [20.0, 20.0, 20.0, 10.0]), x_scale='jac')
    _params = result.x
    return _params

def predict(row) -> float:
    return _forecast(_inputs(row), _params)
