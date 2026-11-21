import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    x = float(row['x_now'])
    u = float(row['u_now'])
    h = float(row['H'])
    z = row.get('mem_conv')
    if z is not None:
        return x, u, h, float(z), None
    prefix = row['raw_prefix']
    t = np.array([point['t'] for point in prefix], dtype=float)
    xs = np.array([point['x'] for point in prefix], dtype=float)
    return x, u, h, None, (t[-1] - t, np.diff(t), xs)


def _forecast(data, params):
    x, u, h, z, history = data
    a, b, c, tau = params
    if z is None:
        ages, widths, xs = history
        weighted = np.exp(-ages / tau) * xs
        z = 0.52 * (weighted[0] + np.dot(widths, weighted[:-1] + weighted[1:]) / (2.0 * tau))
    if h == 0.0:
        return x
    n = int(math.ceil(h / 0.004))
    # Augment the two states with a constant 1 for the affine drive.
    m = (h / n) * np.array([[-a, b, c * u], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    m2 = m @ m
    # Classical RK4 step matrix for the constant linear generator.
    step = np.eye(3) + m + m2 / 2.0 + (m2 @ m) / 6.0 + (m2 @ m2) / 24.0
    return float((np.linalg.matrix_power(step, n) @ np.array([x, z, 1.0]))[0])


def fit(train_rows):
    global _params
    data = [_prepare(row) for row in train_rows]
    target = np.array([row['y'] for row in train_rows], dtype=float)

    def residual(params):
        return np.array([_forecast(row, params) for row in data]) - target

    result = least_squares(residual, [2.0, 1.0, 1.0, 0.7], bounds=([0.0, 0.0, 0.0, 0.01], np.inf), x_scale='jac', gtol=1e-11)
    _params = result.x
    return _params


def predict(row) -> float:
    return _forecast(_prepare(row), _params)
