import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_I = np.eye(3)

def _prepare(row):
    z = row.get('mem_conv')
    prefix = None
    if z is None:
        t, x = np.array([(p['t'], p['x']) for p in row['raw_prefix']], dtype=float).T
        prefix = (np.diff(t), 0.52 * x)
    return float(row['x_now']), float(row['u_now']), float(row['H']), z, prefix

def _forecast(data, p):
    x, u, H, z, prefix = data
    if H == 0:
        return x
    a, b, c, tau = p
    if z is None:
        dt, X = prefix
        z = X[0]
        for d, left, right in zip(dt, X[:-1], X[1:]):
            h = d / tau
            e = math.exp(-h)
            z = e * z + 0.5 * h * (e * left + right)
    # Augment by a constant; the RK4 step is a matrix polynomial.
    n = max(1, math.ceil(H / 0.004))
    B = (H / n) * np.array([[-a, b, c * u], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    R = _I + B @ (_I + B @ (_I / 2 + B @ (_I / 6 + B / 24)))
    return float(np.linalg.matrix_power(R, n)[0] @ np.array([x, z, 1.0]))

def fit(train_rows):
    global _params
    data = [_prepare(row) for row in train_rows]
    y = np.array([row['y'] for row in train_rows], dtype=float)
    def residual(p):
        return np.array([_forecast(d, p) for d in data]) - y
    result = least_squares(residual, [1.4, 1.0, 1.5, 0.4], bounds=([0, 0, 0, 0.01], [np.inf] * 4), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row):
    return _forecast(_prepare(row), _params)
