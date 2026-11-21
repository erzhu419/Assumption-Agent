import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    pref = sorted(pref, key=lambda q: float(q['t']))
    t0, te = float(pref[0]['t']), float(pref[-1]['t'])
    z = 0.52 * float(pref[0]['x']) * math.exp(-(te-t0)/tau)
    for a, b in zip(pref[:-1], pref[1:]):
        ta, tb = float(a['t']), float(b['t'])
        xa, xb = float(a['x']), float(b['x'])
        z += 0.52 * 0.5 * (math.exp(-(te-ta)/tau)*xa + math.exp(-(te-tb)/tau)*xb) * (tb-ta) / tau
    return z

def _integrate(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x+0.5*h*k1x, z+0.5*h*k1z)
        k3x, k3z = f(x+0.5*h*k2x, z+0.5*h*k2z)
        k4x, k4z = f(x+h*k3x, z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6
        z += h*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    global _P
    rows = [r for r in train_rows if r.get('y') is not None]
    if not rows:
        raise ValueError('fit requires labeled training rows')
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_integrate(r, p) for r in rows]) - y
    starts = [[0.5, 0.5, 0.5, 0.2], [1.0, 0.3, 0.8, 0.5], [0.2, 0.8, 0.5, 1.0]]
    sols = [least_squares(residual, s, bounds=([0,0,0,0.01],[5,5,5,5]), max_nfev=3000) for s in starts]
    _P = min(sols, key=lambda q: float(np.dot(q.fun, q.fun))).x
    return _P

def predict(row):
    if _P is None:
        raise ValueError('call fit(train_rows) first')
    return float(_integrate(row, _P))