import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _initial_z(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row['raw_prefix']
    z = 0.52 * float(prefix[0]['x'])
    for left, right in zip(prefix[:-1], prefix[1:]):
        dt = float(right['t']) - float(left['t'])
        decay = math.exp(-dt / tau)
        z = decay * z + 0.52 * dt / (2.0 * tau) * (float(left['x']) * decay + float(right['x']))
    return z

def _predict(row, p):
    a, b, c, tau, vmax = p
    x = float(row['x_now'])
    z = _initial_z(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.004)))
    dt = H / n
    def f(xv, zv):
        return (-a * xv + b * zv + c * u, (0.52 * xv - zv) / tau)
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = f(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6
    s = float(row['s_grad'])
    return x + vmax * (1.0 - 0.70*s/(1.0+s)) * u/(1.2+u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_predict(r, p) - float(r['y']) for r in train_rows])
    starts = [[0.5, 0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0, 0.5], [0.2, 0.3, 0.3, 2.0, 2.0]]
    results = [least_squares(residual, p, bounds=([0, 0, 0, 0.02, 0], [10, 10, 10, 20, 20]), max_nfev=2000) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_predict(row, _params))
