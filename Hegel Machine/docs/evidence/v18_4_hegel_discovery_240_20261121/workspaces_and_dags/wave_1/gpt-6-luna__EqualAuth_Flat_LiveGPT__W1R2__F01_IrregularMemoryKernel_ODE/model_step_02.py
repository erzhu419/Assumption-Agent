import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z_from_prefix(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(pts[0]['x'])
    for p, q in zip(pts[:-1], pts[1:]):
        dt = float(q['t']) - float(p['t'])
        x0, x1 = float(p['x']), float(q['x'])
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        for j in range(n):
            xa = x0 + (x1 - x0) * j / n
            xb = x0 + (x1 - x0) * (j + 1) / n
            xm = 0.5 * (xa + xb)
            k1 = (0.52 * xa - z) / tau
            k2 = (0.52 * xm - (z + 0.5*h*k1)) / tau
            k3 = (0.52 * xm - (z + 0.5*h*k2)) / tau
            k4 = (0.52 * xb - (z + h*k3)) / tau
            z += h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return z

def _simulate(row, pars):
    a, b, c, tau = pars
    x = float(row['x_now'])
    z = _z_from_prefix(row, tau)
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    u = float(row['u_now'])
    def rhs(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx - zz)/tau
    for _ in range(n):
        k1x,k1z = rhs(x,z)
        k2x,k2z = rhs(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z = rhs(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z = rhs(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    def unpack(q):
        return np.exp(q)
    def residual(q):
        pars = unpack(q)
        return np.array([_simulate(r, pars) - float(r['y']) for r in train_rows])
    starts = [np.log([0.5, 0.5, 0.5, 0.5]), np.log([1.0, 0.5, 1.0, 1.0]), np.log([0.2, 1.0, 0.5, 2.0])]
    sols = [least_squares(residual, s, max_nfev=3000) for s in starts]
    best = min(sols, key=lambda s: np.sum(s.fun**2))
    _PARAMS = unpack(best.x)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))