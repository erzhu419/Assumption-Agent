import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    pts = sorted(pts, key=lambda p: p['t'])
    z = 0.52 * float(pts[0]['x'])
    t0 = float(pts[0]['t'])
    for p0, p1 in zip(pts[:-1], pts[1:]):
        dt = float(p1['t']) - float(p0['t'])
        x0, x1 = float(p0['x']), float(p1['x'])
        # Exponential-memory update with trapezoidal forcing on each interval.
        e = math.exp(-dt / tau)
        z = e * z + 0.52 * (x0 + x1) * 0.5 * (1.0 - e)
    return z

def _simulate(row, pars):
    a, b, c, tau = map(float, pars)
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(xv, zv):
        return -a * xv + b * zv + c * u, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*h*k1x, z + 0.5*h*k1z)
        k3x, k3z = f(x + 0.5*h*k2x, z + 0.5*h*k2z)
        k4x, k4z = f(x + h*k3x, z + h*k3z)
        x += h * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += h * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    def residual(p):
        return np.asarray([_simulate(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([0.7, 0.5, 0.6, 0.8]),
                           bounds=([0.001, 0.001, 0.001, 0.02], [5.0, 5.0, 5.0, 10.0]),
                           max_nfev=3000)
    return result.x

def predict(row):
    return float(_simulate(row, _FIT_PARAMS))