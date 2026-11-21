import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    p = row['raw_prefix']
    t_end = float(p[-1]['t'])
    z = 0.52 * float(p[0]['x']) * math.exp(-(t_end - float(p[0]['t'])) / tau)
    for left, right in zip(p[:-1], p[1:]):
        dt = float(right['t']) - float(left['t'])
        f0 = 0.52 * float(left['x']) * math.exp(-(t_end - float(left['t'])) / tau) / tau
        f1 = 0.52 * float(right['x']) * math.exp(-(t_end - float(right['t'])) / tau) / tau
        z += 0.5 * dt * (f0 + f1)
    return z

def _forecast(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _memory(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    best = None
    for start in ([1.0, 0.5, 1.0, 0.2], [2.0, 1.0, 2.0, 0.5], [1.0, 1.0, 1.0, 1.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 0.0, 0.03], [20.0, 20.0, 20.0, 10.0]), max_nfev=2000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_forecast(row, _params))