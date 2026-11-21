import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _z0(row, tau):
    pref = row.get('raw_prefix')
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(q['t']) for q in pref], dtype=float)
    xs = np.array([float(q['x']) for q in pref], dtype=float)
    z = 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        n = max(1, int(math.ceil(dt / 0.002)))
        h = dt / n
        for j in range(n):
            f0 = j / n
            f1 = (j + 1) / n
            xm = xs[i-1] + (xs[i] - xs[i-1]) * (f0 + f1) * 0.5
            z += h * (0.52 * xm - z) / tau
    return z

def _pred(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def rhs(xx, zz):
        return -a * xx + b * zz + c * u, (0.52 * xx - zz) / tau
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = rhs(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = rhs(x + h*k3x, z + h*k3z)
        x += h * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += h * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return x

def fit(train_rows):
    global _P
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - ys
    best = None
    for start in ([0.2,0.2,0.5,0.2], [0.5,0.5,0.5,0.5], [1.0,0.2,0.2,0.1], [0.1,1.0,0.5,1.0]):
        result = least_squares(residual, start, bounds=([0,0,0,0.005],[5,5,5,5]), max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _P = best.x
    return _P

def predict(row) -> float:
    return float(_pred(row, _P))