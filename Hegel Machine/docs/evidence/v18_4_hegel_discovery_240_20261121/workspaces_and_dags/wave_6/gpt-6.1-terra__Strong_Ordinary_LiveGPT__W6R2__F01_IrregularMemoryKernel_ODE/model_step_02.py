import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict(row, p):
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
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.004)))
    h = H / n
    def rhs(x, z):
        return -a*x + b*z + c*u, (0.52*x - z)/tau
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = rhs(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = rhs(x + h*k3x, z + h*k3z)
        x += h*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += h*(k1z + 2*k2z + 2*k3z + k4z)/6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_predict(r, p) - float(r['y']) for r in train_rows])
    best = None
    for tau0 in (0.15, 0.5, 1.0, 2.0):
        result = least_squares(residual, [1.0, 0.7, 1.0, tau0], bounds=([0.0, 0.0, 0.0, 0.03], [10.0, 10.0, 10.0, 10.0]), max_nfev=1500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return _predict(row, _params)
