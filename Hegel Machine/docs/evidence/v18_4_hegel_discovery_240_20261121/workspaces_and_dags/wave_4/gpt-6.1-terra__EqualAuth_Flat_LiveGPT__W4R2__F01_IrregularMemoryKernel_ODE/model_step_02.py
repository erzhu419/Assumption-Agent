import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, c, tau = p
    if row.get('mem_conv') is not None:
        z = float(row['mem_conv'])
    else:
        prefix = row['raw_prefix']
        z = 0.52 * float(prefix[0]['x'])
        for left, right in zip(prefix[:-1], prefix[1:]):
            dt = float(right['t']) - float(left['t'])
            decay = math.exp(-dt / tau)
            z = decay * z + 0.52 * dt / (2.0 * tau) * (decay * float(left['x']) + float(right['x']))
    x = float(row['x_now'])
    u = float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, int(math.ceil(horizon / 0.004)))
    dt = horizon / n
    def f(xv, zv):
        return -a * xv + b * zv + c * u, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = f(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    best = None
    for tau in (0.15, 0.5, 1.5):
        result = least_squares(residual, [1.0, 0.5, 1.0, tau], bounds=([0.0, 0.0, 0.0, 0.03], [10.0, 10.0, 10.0, 10.0]), max_nfev=500)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_forecast(row, _params))