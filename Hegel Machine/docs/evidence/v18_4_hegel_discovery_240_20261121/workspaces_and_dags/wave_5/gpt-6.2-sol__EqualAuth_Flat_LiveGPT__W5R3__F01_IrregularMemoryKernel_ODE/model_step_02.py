import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    x = float(row['x_now'])
    z = row.get('mem_conv')
    prefix = row.get('raw_prefix')
    if z is not None:
        memory = (float(z), None, None)
    elif prefix:
        times = np.array([p['t'] for p in prefix], dtype=float)
        values = np.array([p['x'] for p in prefix], dtype=float)
        memory = (0.52 * values[0], times[-1] - times, values)
    else:
        memory = (0.52 * x, None, None)
    return x, float(row['u_now']), float(row['H']), memory

def _forecast(row, params):
    a, b, c, tau = map(float, params)
    x, u, horizon, (z, lag, values) = row
    if lag is not None:
        integrand = (0.52 / tau) * values * np.exp(-lag / tau)
        z = z * math.exp(-lag[0] / tau) + 0.5 * float(np.dot(lag[:-1] - lag[1:], integrand[:-1] + integrand[1:]))
    if horizon == 0:
        return x
    n = max(1, math.ceil(horizon / 0.004))
    h = horizon / n
    hh = h / 2
    drive = c * u
    inv_tau = 1 / tau
    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) * inv_tau
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + hh*k1, z + hh*l1)
        k3, l3 = rhs(x + hh*k2, z + hh*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        z += (h/6) * (l1 + 2*l2 + 2*l3 + l4)
    return float(x)

def fit(train_rows):
    global _params
    rows = [_prepare(row) for row in train_rows]
    targets = np.array([row['y'] for row in train_rows], dtype=float)
    def residual(params):
        return np.array([_forecast(row, params) for row in rows]) - targets
    result = least_squares(residual, [1.4, 1.0, 1.4, 0.3], bounds=([0, 0, 0, 0.015], [np.inf]*4), x_scale='jac')
    _params = result.x

def predict(row):
    return _forecast(_prepare(row), _params)