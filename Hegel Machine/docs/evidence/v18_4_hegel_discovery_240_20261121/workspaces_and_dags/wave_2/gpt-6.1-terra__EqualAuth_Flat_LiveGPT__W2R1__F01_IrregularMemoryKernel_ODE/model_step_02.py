import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _initial_memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row['raw_prefix']
    latest = float(prefix[-1]['t'])
    z = 0.52 * float(prefix[0]['x']) * math.exp(-(latest - float(prefix[0]['t'])) / tau)
    integral = 0.0
    for left, right in zip(prefix[:-1], prefix[1:]):
        t0, t1 = float(left['t']), float(right['t'])
        f0 = float(left['x']) * math.exp(-(latest - t0) / tau)
        f1 = float(right['x']) * math.exp(-(latest - t1) / tau)
        integral += 0.5 * (t1 - t0) * (f0 + f1)
    return z + 0.52 * integral / tau

def _forecast(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _initial_memory(row, tau)
    u = float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, int(math.ceil(abs(horizon) / 0.004)))
    dt = horizon / n
    def derivative(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = derivative(x, z)
        k2x, k2z = derivative(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = derivative(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = derivative(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(row, p) - float(row['y']) for row in train_rows])
    result = least_squares(residual, [1.0, 0.5, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.03], [20.0, 20.0, 20.0, 20.0]), max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return _forecast(row, _params)
