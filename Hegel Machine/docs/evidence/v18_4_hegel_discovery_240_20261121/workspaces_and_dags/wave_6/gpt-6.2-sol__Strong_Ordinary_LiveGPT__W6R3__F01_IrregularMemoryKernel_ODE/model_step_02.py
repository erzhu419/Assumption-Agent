import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_I = np.eye(3)


def _prepare(row):
    prefix = row.get('raw_prefix')
    if prefix:
        t = np.array([p['t'] for p in prefix], dtype=float)
        xs = np.array([p['x'] for p in prefix], dtype=float)
        history = (np.diff(t), xs)
    else:
        history = None
    return float(row['u_now']), float(row['x_now']), float(row['H']), row.get('mem_conv'), history


def _forecast(p, data):
    a, b, c, tau = p
    u, x, H, z, history = data
    if H == 0:
        return x
    if z is None:
        if history is None:
            z = 0.52 * x
        else:
            # Integrate the exponential kernel exactly on each linear segment.
            intervals, xs = history
            z = 0.52 * xs[0]
            for dt, left, right in zip(intervals, xs[:-1], xs[1:]):
                if dt == 0:
                    continue
                q = dt / tau
                w = -math.expm1(-q)
                z = (1 - w) * z + 0.52 * (w * left + (1 - w / q) * (right - left))
    # With constant forcing, repeated RK4 steps are powers of one matrix.
    n = max(1, math.ceil(H / 0.004))
    B = (H / n) * np.array([[-a, b, c * u], [0.52 / tau, -1 / tau, 0.0], [0.0, 0.0, 0.0]])
    step = _I + B @ (_I + B @ (_I / 2 + B @ (_I / 6 + B / 24)))
    return float(np.linalg.matrix_power(step, n)[0] @ np.array([x, z, 1.0]))


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = [_prepare(row) for row in rows]
    y = np.array([row['y'] for row in rows], dtype=float)
    def residual(p):
        return np.array([_forecast(p, item) for item in data]) - y
    _params = least_squares(residual, [2.0, 1.4, 1.4, 0.5], bounds=([0.0, 0.0, 0.0, 0.02], [15.0, 15.0, 15.0, 5.0]), ftol=1e-11, xtol=1e-11, gtol=1e-11).x


def predict(row) -> float:
    return _forecast(_params, _prepare(row))
