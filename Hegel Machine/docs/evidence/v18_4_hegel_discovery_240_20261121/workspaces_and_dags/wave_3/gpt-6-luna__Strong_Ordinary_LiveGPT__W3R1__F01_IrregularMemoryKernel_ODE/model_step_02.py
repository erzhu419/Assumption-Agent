import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return 0.52 * float(row['x_now'])
    prefix = sorted(prefix, key=lambda q: q['t'])
    z = 0.52 * float(prefix[0]['x'])
    t0 = float(prefix[0]['t'])
    for left, right in zip(prefix[:-1], prefix[1:]):
        dt = float(right['t']) - float(left['t'])
        if dt <= 0:
            continue
        decay = math.exp(-dt / tau)
        # Trapezoidal exponential-kernel update for linearly varying x.
        xmid = 0.5 * (float(left['x']) + float(right['x']))
        z = decay * z + 0.52 * xmid * (1.0 - decay)
        t0 = float(right['t'])
    return z

def _predict(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(xv, zv):
        return -a * xv + b * zv + c * u, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        z += (dt/6.0) * (k1z + 2*k2z + 2*k3z + k4z)
    return x

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict(r, p) for r in train_rows]) - ys
    starts = ([0.4, 0.2, 0.3, 0.1], [1.0, 0.5, 0.8, 0.2], [0.2, 0.8, 0.5, 0.05])
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([1e-5, 1e-5, 1e-5, 1e-4], [10.0, 10.0, 10.0, 10.0]), max_nfev=3000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    return float(_predict(row, fit([dict(row, y=0.0)])))