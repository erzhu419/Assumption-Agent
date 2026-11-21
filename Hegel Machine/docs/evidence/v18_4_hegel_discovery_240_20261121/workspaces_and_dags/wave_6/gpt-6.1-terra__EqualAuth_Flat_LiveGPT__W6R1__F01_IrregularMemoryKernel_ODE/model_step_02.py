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
        z = 0.52 * float(prefix[0]['x'])
        for left, right in zip(prefix[:-1], prefix[1:]):
            dt = float(right['t']) - float(left['t'])
            decay = math.exp(-dt / tau)
            z = decay * z + 0.52 * dt / (2.0 * tau) * (float(left['x']) * decay + float(right['x']))
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def deriv(x, z):
        return -a*x + b*z + c*u, (0.52*x - z)/tau
    for _ in range(n):
        k1, l1 = deriv(x, z)
        k2, l2 = deriv(x + h*k1/2, z + h*l1/2)
        k3, l3 = deriv(x + h*k2/2, z + h*l2/2)
        k4, l4 = deriv(x + h*k3, z + h*l3)
        x += h*(k1 + 2*k2 + 2*k3 + k4)/6
        z += h*(l1 + 2*l2 + 2*l3 + l4)/6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    best = None
    for tau0 in (0.15, 0.5, 1.5):
        result = least_squares(residual, [1.0, 0.5, 1.0, tau0], bounds=([0.0, -5.0, -5.0, 0.03], [10.0, 10.0, 10.0, 10.0]), max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _forecast(row, _params)
