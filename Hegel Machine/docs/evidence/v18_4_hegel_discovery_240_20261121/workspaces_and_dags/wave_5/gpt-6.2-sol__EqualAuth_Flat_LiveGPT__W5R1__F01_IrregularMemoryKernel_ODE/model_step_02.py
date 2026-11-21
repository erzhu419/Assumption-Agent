import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    mem = row.get('mem_conv')
    if mem is not None:
        return float(mem)
    prefix = row['raw_prefix']
    z = 0.52 * prefix[0]['x']
    for left, right in zip(prefix, prefix[1:]):
        q = (right['t'] - left['t']) / tau
        decay = math.exp(-q)
        z = decay * z + 0.26 * q * (decay * left['x'] + right['x'])
    return z

def _forecast(row, params):
    a, b, c, tau = map(float, params)
    x = float(row['x_now'])
    z = _memory(row, tau)
    drive = c * float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, math.ceil(horizon / 0.004))
    h = horizon / n
    half = 0.5 * h
    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z)/tau
    for _ in range(n):
        k1, j1 = rhs(x, z)
        k2, j2 = rhs(x + half*k1, z + half*j1)
        k3, j3 = rhs(x + half*k2, z + half*j2)
        k4, j4 = rhs(x + h*k3, z + h*j3)
        x += (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        z += (h/6.0) * (j1 + 2*j2 + 2*j3 + j4)
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(params):
        return np.array([_forecast(r, params) for r in rows]) - y
    _params = least_squares(residual, [1.5, 1.0, 1.5, 0.6], bounds=([0.0, 0.0, 0.0, 0.02], [20.0, 20.0, 20.0, 10.0]), x_scale='jac').x

def predict(row) -> float:
    return _forecast(row, _params)
