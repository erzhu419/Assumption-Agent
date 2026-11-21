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
        z = decay * z + 0.52 * dt / (2.0 * tau) * (float(left['x']) * decay + float(right['x']))
    return z

def _model(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x = float(row['x_now'])
    z = _initial_memory(row, tau)
    horizon = float(row['H'])
    n = max(1, int(math.ceil(horizon / 0.004)))
    dt = horizon / n
    def deriv(xv, zv):
        return -a * xv + b * zv + drive, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = deriv(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = deriv(x + dt * k3x, z + dt * k3z)
        x += dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        z += dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_model(row, p) - float(row['y']) for row in rows])
    lower = [0.0, 0.0, 1.0, 0.05, 0.03]
    upper = [5.0, 5.0, 15.0, 0.95, 5.0]
    starts = [[0.7, 0.5, 6.0, 0.7, 0.3], [1.0, 1.0, 7.0, 0.7, 1.0], [0.3, 0.3, 5.0, 0.7, 0.1]]
    results = [least_squares(residual, p, bounds=(lower, upper), max_nfev=2000, xtol=1e-11, ftol=1e-11, gtol=1e-11) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return _model(row, _params)
