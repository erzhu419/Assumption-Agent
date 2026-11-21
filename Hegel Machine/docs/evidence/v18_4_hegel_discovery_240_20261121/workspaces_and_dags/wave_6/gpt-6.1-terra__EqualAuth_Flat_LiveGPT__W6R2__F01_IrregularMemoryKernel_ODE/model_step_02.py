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
        t_end = float(prefix[-1]['t'])
        z = 0.52 * float(prefix[0]['x']) * math.exp(-(t_end - float(prefix[0]['t'])) / tau)
        for left, right in zip(prefix[:-1], prefix[1:]):
            t0, t1 = float(left['t']), float(right['t'])
            f0 = 0.52 * float(left['x']) * math.exp(-(t_end - t0) / tau) / tau
            f1 = 0.52 * float(right['x']) * math.exp(-(t_end - t1) / tau) / tau
            z += 0.5 * (t1 - t0) * (f0 + f1)
    x = float(row['x_now'])
    u = float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, int(math.ceil(horizon / 0.004)))
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
    return x

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(row, p) - float(row['y']) for row in train_rows])
    best = None
    for start in ([1.5, 0.8, 1.0, 0.5], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 1.0, 0.2]):
        result = least_squares(residual, start, bounds=([0.0, -10.0, 0.0, 0.03], [10.0, 10.0, 10.0, 10.0]), max_nfev=1200, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_forecast(row, _params))