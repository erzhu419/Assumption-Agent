import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _memory_init(row, tau):
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in prefix], dtype=float)
    xs = np.array([float(p['x']) for p in prefix], dtype=float)
    z = 0.52 * xs[0]
    for i in range(len(ts) - 1):
        dt = ts[i + 1] - ts[i]
        if dt <= 0:
            continue
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        for j in range(n):
            f0 = j / n
            f1 = (j + 1) / n
            xm = xs[i] + (xs[i + 1] - xs[i]) * (f0 + f1) * 0.5
            z += h * (0.52 * xm - z) / tau
    return z

def _sim(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _memory_init(row, tau) if row.get('mem_conv') is None else float(row['mem_conv'])
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
        x += h*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += h*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_sim(r, p) for r in train_rows]) - y
    starts = [np.array([0.5, 0.2, 0.5, 0.2]), np.array([1.0, 0.1, 1.0, 0.5]), np.array([0.2, 0.3, 0.3, 1.0])]
    sols = [least_squares(residual, s, bounds=([0, 0, -10, 0.005], [10, 10, 10, 10]), max_nfev=3000) for s in starts]
    _PARAMS = min(sols, key=lambda q: np.sum(q.fun*q.fun)).x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit must be called first')
    return float(_sim(row, _PARAMS))