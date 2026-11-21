import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _initial_memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row['raw_prefix']
    z = 0.52 * float(prefix[0]['x'])
    for left, right in zip(prefix[:-1], prefix[1:]):
        dt = float(right['t']) - float(left['t'])
        decay = math.exp(-dt / tau)
        z = decay * z + 0.52 * dt * (decay * float(left['x']) + float(right['x'])) / (2.0 * tau)
    return z

def _forecast(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _initial_memory(row, tau)
    u = float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, int(math.ceil(abs(horizon) / 0.004)))
    dt = horizon / n
    def deriv(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(row, p) - float(row['y']) for row in train_rows])
    best = None
    for start in ([1.0, 0.5, 1.0, 0.5], [2.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 0.15]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 0.0, 0.03], [20.0, 30.0, 20.0, 20.0]), max_nfev=1000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return _forecast(row, _params)
