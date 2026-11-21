import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row['raw_prefix']
    z = 0.52 * float(prefix[0]['x'])
    for left, right in zip(prefix[:-1], prefix[1:]):
        dt = float(right['t']) - float(left['t'])
        decay = math.exp(-dt / tau)
        z = decay * z + 0.52 * dt / (2.0 * tau) * (decay * float(left['x']) + float(right['x']))
    return z

def _forecast(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _memory(row, tau)
    u = float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, int(math.ceil(abs(horizon) / 0.004)))
    dt = horizon / n
    def deriv(x, z):
        return -a * x + b * z + c * u, (0.52 * x - z) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt * k1x / 2, z + dt * k1z / 2)
        k3x, k3z = deriv(x + dt * k2x / 2, z + dt * k2z / 2)
        k4x, k4z = deriv(x + dt * k3x, z + dt * k3z)
        x += dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        z += dt * (k1z + 2 * k2z + 2 * k3z + k4z) / 6
    return x

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, [1.0, 0.5, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.03], [10.0, 10.0, 10.0, 10.0]), max_nfev=1200, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row):
    return float(_forecast(row, _params))