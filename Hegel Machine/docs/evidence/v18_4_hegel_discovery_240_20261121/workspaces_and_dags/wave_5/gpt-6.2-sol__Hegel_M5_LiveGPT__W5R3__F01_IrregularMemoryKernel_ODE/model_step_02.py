import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    z = row.get('mem_conv')
    history = ()
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * prefix[0]['x']
        history = tuple((hi['t'] - lo['t'], lo['x'], hi['x'] - lo['x'])
                        for lo, hi in zip(prefix, prefix[1:]))
    return row['x_now'], z, row['u_now'], row['H'], history

def _forecast(p, data):
    a, b, c, tau = p
    x, z, u, H, history = data
    # Exact memory update on each interval with linearly interpolated x.
    for dt, left_x, delta_x in history:
        q = dt / tau
        weight = -math.expm1(-q)
        z = math.exp(-q) * z + 0.52 * (weight * left_x + (1.0 - weight / q) * delta_x)

    n = max(1, math.ceil(H / 0.004))
    h = H / n
    drive = c * u
    inv_tau = 1.0 / tau

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) * inv_tau

    for _ in range(n):
        k1, j1 = rhs(x, z)
        k2, j2 = rhs(x + 0.5*h*k1, z + 0.5*h*j1)
        k3, j3 = rhs(x + 0.5*h*k2, z + 0.5*h*j2)
        k4, j4 = rhs(x + h*k3, z + h*j3)
        x += h/6 * (k1 + 2*k2 + 2*k3 + k4)
        z += h/6 * (j1 + 2*j2 + 2*j3 + j4)
    return float(x)

def fit(train_rows):
    global _params
    data = [_prepare(row) for row in train_rows]
    y = np.array([row['y'] for row in train_rows], dtype=float)

    def residuals(p):
        return np.array([_forecast(p, d) for d in data]) - y

    _params = least_squares(residuals, [1.5, 1.0, 1.5, 0.5],
                            bounds=([0.0, 0.0, 0.0, 0.01], np.inf),
                            ftol=1e-10, xtol=1e-10, gtol=1e-10).x
    return _params

def predict(row) -> float:
    return _forecast(_params, _prepare(row))