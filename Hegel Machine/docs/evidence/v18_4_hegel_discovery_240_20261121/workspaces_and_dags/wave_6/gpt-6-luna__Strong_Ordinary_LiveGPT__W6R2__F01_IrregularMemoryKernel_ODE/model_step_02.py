import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    hist = row.get('raw_prefix') or []
    if not hist:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(hist[0]['x'])
    t_prev = float(hist[0]['t'])
    x_prev = float(hist[0]['x'])
    for q in hist[1:]:
        t1, x1 = float(q['t']), float(q['x'])
        d = t1 - t_prev
        n = max(1, int(math.ceil(d / 0.004)))
        h = d / n
        for i in range(n):
            xa = x_prev + (x1 - x_prev) * i / n
            xb = x_prev + (x1 - x_prev) * (i + 1) / n
            xm = 0.5 * (xa + xb)
            z += h * (0.52 * xm - z) / tau
        t_prev, x_prev = t1, x1
    return z

def _simulate(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(xx, zz):
        return (-a * xx + b * zz + c * u, (0.52 * xx - zz) / tau)
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
        return np.array([_simulate(r, p) - float(r['y']) for r in train_rows])
    starts = [np.array([0.5, 0.5, 0.5, 0.1]), np.array([1.0, 0.2, 0.8, 0.2]), np.array([0.2, 1.0, 0.3, 0.05])]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.001, 0.001, 0.001, 0.005], [5.0, 5.0, 5.0, 2.0]), max_nfev=500)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    return float(_simulate(row, _PARAMS))