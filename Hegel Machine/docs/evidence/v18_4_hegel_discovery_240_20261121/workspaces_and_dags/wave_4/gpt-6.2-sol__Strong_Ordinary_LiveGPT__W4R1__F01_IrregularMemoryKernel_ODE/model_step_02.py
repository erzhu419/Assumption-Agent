import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    z = row.get('mem_conv')
    prefix = ()
    if z is None:
        points = sorted(row['raw_prefix'], key=lambda p: p['t'])
        prefix = tuple((p['t'] - points[0]['t'], p['x']) for p in points)
    h = row['H']
    n = max(1, math.ceil(h / 0.004))
    return row['x_now'], row['u_now'], z, prefix, n, h / n

def _forecast(params, data):
    a, b, c, tau = map(float, params)
    x, u, z, prefix, n, dt = data
    if z is None:
        z = 0.52 * prefix[0][1]
        for (t0, x0), (t1, x1) in zip(prefix, prefix[1:]):
            q = (t1 - t0) / tau
            e = math.exp(-q)
            z = e * z + 0.26 * q * (e * x0 + x1)
    drive = c * u
    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z) / tau
    half = dt / 2
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + half*k1, z + half*l1)
        k3, l3 = rhs(x + half*k2, z + half*l2)
        k4, l4 = rhs(x + dt*k3, z + dt*l3)
        x += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        z += dt/6 * (l1 + 2*l2 + 2*l3 + l4)
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = [_prepare(row) for row in rows]
    y = np.array([row['y'] for row in rows], dtype=float)
    def residual(params):
        return np.array([_forecast(params, d) for d in data]) - y
    _params = least_squares(residual, [1.5, 1.0, 1.5, 0.4], bounds=([0, 0, 0, 0.025], [20, 20, 20, 10]), x_scale='jac').x

def predict(row):
    return _forecast(_params, _prepare(row))