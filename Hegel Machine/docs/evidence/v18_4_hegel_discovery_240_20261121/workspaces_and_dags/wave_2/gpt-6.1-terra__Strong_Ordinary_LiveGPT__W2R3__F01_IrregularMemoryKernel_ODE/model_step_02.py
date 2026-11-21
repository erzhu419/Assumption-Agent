import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row['raw_prefix']
    z = 0.52 * float(prefix[0]['x'])
    for p, q in zip(prefix[:-1], prefix[1:]):
        dt = float(q['t']) - float(p['t'])
        decay = math.exp(-dt / tau)
        z = decay * z + 0.52 * dt / (2.0 * tau) * (float(p['x']) * decay + float(q['x']))
    return z

def _predict(row, params):
    a, b, c, tau = params
    x = float(row['x_now'])
    z = _memory(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.004)))
    dt = H / n
    def deriv(x, z):
        return (-a*x + b*z + c*u, (0.52*x - z)/tau)
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
    rows = list(train_rows)
    def residual(p):
        return np.array([_predict(r, p) - float(r['y']) for r in rows])
    best = None
    for tau in (0.1, 0.3, 0.8, 2.0):
        result = least_squares(residual, [1.0, 1.0, 1.0, tau], bounds=([0.0, -10.0, -10.0, 0.03], [10.0, 10.0, 10.0, 10.0]), max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _predict(row, _params)
