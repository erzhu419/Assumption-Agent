import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    prefix = row.get('raw_prefix')
    if not prefix:
        return 0.52 * float(row['x_now'])
    pts = sorted(prefix, key=lambda p: p['t'])
    z = 0.52 * float(pts[0]['x'])
    for a, b in zip(pts[:-1], pts[1:]):
        dt = float(b['t']) - float(a['t'])
        decay = math.exp(-dt / tau)
        xmid = 0.5 * (float(a['x']) + float(b['x']))
        z = z * decay + 0.52 * xmid * (1.0 - decay)
    return z

def _simulate(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xx, zz):
        return -a * xx + b * zz + c * u, (0.52 * xx - zz) / tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array([0.5, 0.2, 0.5, 0.2]), np.array([1.0, 0.5, 1.0, 0.5]), np.array([0.2, 0.1, 0.2, 0.1])]
    sols = [least_squares(residual, p, bounds=([0.0, -5.0, -5.0, 0.005], [10.0, 5.0, 10.0, 5.0]), max_nfev=3000) for p in starts]
    return min(sols, key=lambda s: float(np.dot(s.fun, s.fun))).x

def predict(row):
    return float(_simulate(row, _FIT_PARAMS))