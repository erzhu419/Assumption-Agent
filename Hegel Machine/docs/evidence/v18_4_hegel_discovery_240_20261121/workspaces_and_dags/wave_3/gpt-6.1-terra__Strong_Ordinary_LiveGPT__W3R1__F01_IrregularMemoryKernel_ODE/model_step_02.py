import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, c, tau = p
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * prefix[0]['x']
        for left, right in zip(prefix[:-1], prefix[1:]):
            dt = right['t'] - left['t']
            decay = math.exp(-dt / tau)
            z = decay * z + 0.52 * dt * (left['x'] * decay + right['x']) / (2.0 * tau)
    x = float(row['x_now'])
    u = float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, int(math.ceil(abs(horizon) / 0.004)))
    h = horizon / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x - z)/tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = f(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = f(x + h*k3x, z + h*k3z)
        x += h*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += h*(k1z + 2*k2z + 2*k3z + k4z)/6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(row, p) - row['y'] for row in train_rows])
    bounds = ([0.0, 0.0, 0.0, 0.03], [10.0, 10.0, 10.0, 10.0])
    starts = [[1.0, 0.5, 1.0, 0.3], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 1.0, 2.0]]
    fits = [least_squares(residual, p, bounds=bounds, max_nfev=1000) for p in starts]
    _params = min(fits, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return _forecast(row, _params)
