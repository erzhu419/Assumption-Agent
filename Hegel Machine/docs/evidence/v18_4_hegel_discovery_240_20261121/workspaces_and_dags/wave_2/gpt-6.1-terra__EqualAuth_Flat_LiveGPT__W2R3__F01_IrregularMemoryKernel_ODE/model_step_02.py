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
            z = z * decay + 0.52 * dt / (2.0 * tau) * (left['x'] * decay + right['x'])
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def deriv(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x+h*k1x/2, z+h*k1z/2)
        k3x, k3z = deriv(x+h*k2x/2, z+h*k2z/2)
        k4x, k4z = deriv(x+h*k3x, z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6
        z += h*(k1z+2*k2z+2*k3z+k4z)/6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(r, p)-r['y'] for r in train_rows])
    best = None
    for tau in (0.15, 0.5, 1.5, 4.0):
        result = least_squares(residual, [1.0, 0.5, 1.0, tau], bounds=([0.0, 0.0, 0.0, 0.02], [10.0, 10.0, 10.0, 20.0]), max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _forecast(row, _params)
