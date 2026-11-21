import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return float(row.get('mem_conv') or 0.52 * row['x_now'])
    pts = sorted(prefix, key=lambda p: p['t'])
    z = 0.52 * float(pts[0]['x'])
    for p0, p1 in zip(pts[:-1], pts[1:]):
        dt = float(p1['t']) - float(p0['t'])
        if dt <= 0:
            continue
        e = math.exp(-dt / tau)
        x0, x1 = float(p0['x']), float(p1['x'])
        slope = (x1 - x0) / dt
        z = e*z + 0.52*(x0*(1.0-e) + slope*(dt - tau*(1.0-e)))
    return z

def _pred(row, pars):
    a, b, c, tau = map(float, pars)
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=f(x,z); k2x,k2z=f(x+h*k1x/2,z+h*k1z/2)
        k3x,k3z=f(x+h*k2x/2,z+h*k2z/2); k4x,k4z=f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6
        z += h*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    global _PARAMS
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - ys
    best = None
    for start in ([0.3,0.5,0.5,0.2], [0.8,0.2,0.8,0.5], [0.1,0.8,0.3,0.08]):
        res = least_squares(residual, start, bounds=([0.001,-5,-5,0.005],[5,5,5,3]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_pred(row, _PARAMS))