import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _initial_z(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row.get('raw_prefix')
    if not prefix:
        raise ValueError('mem_conv or raw_prefix is required')
    z = 0.52 * float(prefix[0]['x'])
    for left, right in zip(prefix[:-1], prefix[1:]):
        dt = float(right['t']) - float(left['t'])
        decay = math.exp(-dt / tau)
        z = decay * z + 0.26 * (dt / tau) * (float(left['x']) * decay + float(right['x']))
    return z

def _predict(row, p):
    a, b, c, tau, vmax = p
    x = float(row['x_now'])
    z = _initial_z(row, tau)
    u = float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, int(math.ceil(abs(horizon) / 0.004)))
    dt = horizon / n
    def deriv(xv, zv):
        return -a * xv + b * zv + c * u, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6
    s = float(row['s_grad'])
    base = vmax * (1 - 0.705*s/(1+s)) * u/(1.2+u)
    return x + base

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_predict(r, p) - float(r['y']) for r in train_rows])
    best = None
    for start in ([0.5, 0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 0.3, 1.0, 1.0], [0.2, 0.2, 0.2, 0.2, 0.5]):
        result = least_squares(residual, start, bounds=([0, -10, -10, 0.03, -10], [10, 10, 10, 10, 10]), max_nfev=1200)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called first')
    return float(_predict(row, _params))