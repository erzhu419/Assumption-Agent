import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    mem = row.get('mem_conv')
    if mem is not None:
        return float(mem)
    prefix = row.get('raw_prefix')
    if not prefix:
        return 0.52 * float(row['x_now'])
    prefix = sorted(prefix, key=lambda p: p['t'])
    z = 0.52 * float(prefix[0]['x'])
    for left, right in zip(prefix, prefix[1:]):
        q = (float(right['t']) - float(left['t'])) / tau
        decay = math.exp(-q)
        z = decay * z + 0.26 * q * (decay * float(left['x']) + float(right['x']))
    return z

def _forecast(row, params):
    a, b, c, tau = map(float, params)
    x = float(row['x_now'])
    H = float(row['H'])
    z = _memory(row, tau)
    drive = c * float(row['u_now'])
    n = max(1, math.ceil(H / 0.004))
    dt = H / n

    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z) / tau

    for _ in range(n):
        kx1, kz1 = rhs(x, z)
        kx2, kz2 = rhs(x + 0.5*dt*kx1, z + 0.5*dt*kz1)
        kx3, kz3 = rhs(x + 0.5*dt*kx2, z + 0.5*dt*kz2)
        kx4, kz4 = rhs(x + dt*kx3, z + dt*kz3)
        x += dt * (kx1 + 2*kx2 + 2*kx3 + kx4) / 6.0
        z += dt * (kz1 + 2*kz2 + 2*kz3 + kz4) / 6.0
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])

    def residual(params):
        return np.array([_forecast(row, params) for row in rows]) - y

    result = least_squares(residual, [2.0, 1.0, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.01], [np.inf]*4), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row) -> float:
    return _forecast(row, _params)
