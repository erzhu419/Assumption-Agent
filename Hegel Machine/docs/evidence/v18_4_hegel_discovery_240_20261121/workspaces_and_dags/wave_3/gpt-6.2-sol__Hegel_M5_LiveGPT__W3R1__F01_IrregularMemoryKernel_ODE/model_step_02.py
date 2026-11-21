import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    z = row.get('mem_conv')
    history = None
    if z is None:
        prefix = row['raw_prefix']
        t = np.array([p['t'] for p in prefix], dtype=float)
        x = np.array([p['x'] for p in prefix], dtype=float)
        dt = np.diff(t)
        weights = np.zeros_like(t)
        weights[:-1] += 0.5 * dt
        weights[1:] += 0.5 * dt
        history = (t - t[-1], weights * x, x[0])
    H = float(row['H'])
    n = max(1, math.ceil(H / 0.004))
    return float(row['x_now']), float(row['u_now']), z, history, n, H / n

def _forecast(data, params):
    a, b, c, tau = map(float, params)
    x, u, z, history, n, h = data
    if z is None:
        ages, weighted_x, x0 = history
        decay = np.exp(ages / tau)
        z = float(0.52 * (decay[0] * x0 + np.dot(decay, weighted_x) / tau))
    drive = c * u
    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z)/tau
    hh = 0.5*h
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + hh*k1, z + hh*l1)
        k3, l3 = rhs(x + hh*k2, z + hh*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += h*(k1 + 2*k2 + 2*k3 + k4)/6
        z += h*(l1 + 2*l2 + 2*l3 + l4)/6
    return float(x)

def fit(train_rows):
    global _params
    rows = [(_prepare(row), float(row['y'])) for row in train_rows]
    def residuals(params):
        return np.array([_forecast(data, params) - y for data, y in rows])
    _params = least_squares(residuals, [1.6, 1.0, 1.2, 0.5], bounds=([0, 0, 0, 0.02], [np.inf, np.inf, np.inf, np.inf]), x_scale='jac').x

def predict(row):
    return _forecast(_prepare(row), _params)
