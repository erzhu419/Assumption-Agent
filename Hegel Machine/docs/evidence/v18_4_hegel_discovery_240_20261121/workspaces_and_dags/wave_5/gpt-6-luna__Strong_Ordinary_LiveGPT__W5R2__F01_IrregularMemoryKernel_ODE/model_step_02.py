import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(prefix[0]['x'])
    tprev = float(prefix[0]['t'])
    xprev = float(prefix[0]['x'])
    for p in prefix[1:]:
        t = float(p['t']); x = float(p['x'])
        dt = t - tprev
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        for j in range(n):
            xa = xprev + (x - xprev) * (j / n)
            xb = xprev + (x - xprev) * ((j + 1) / n)
            xm = 0.5 * (xa + xb)
            k1 = (0.52 * xa - z) / tau
            k2 = (0.52 * xm - (z + 0.5*h*k1)) / tau
            k3 = (0.52 * xm - (z + 0.5*h*k2)) / tau
            k4 = (0.52 * xb - (z + h*k3)) / tau
            z += h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        tprev, xprev = t, x
    return z

def _simulate(row, p):
    a, b, c, tau = [float(v) for v in p]
    H = float(row['H']); x = float(row['x_now']); z = _z0(row, tau)
    u = float(row['u_now'])
    n = max(1, int(math.ceil(H / 0.004))); h = H / n
    def f(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx - zz)/tau
    for _ in range(n):
        k1x,k1z = f(x,z)
        k2x,k2z = f(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z = f(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z = f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    def unpack(q):
        return np.array([math.exp(q[0]), q[1], q[2], math.exp(q[3])])
    def residual(q):
        p = unpack(q)
        return np.array([_simulate(r,p)-float(r['y']) for r in train_rows])
    starts = [np.array([math.log(0.3), 0.1, 0.5, math.log(0.2)]),
              np.array([math.log(0.7), 0.3, 0.8, math.log(0.5)]),
              np.array([math.log(0.15), -0.1, 0.4, math.log(0.1)])]
    best = None
    for q0 in starts:
        res = least_squares(residual, q0, max_nfev=3000)
        if best is None or np.dot(res.fun,res.fun) < np.dot(best.fun,best.fun):
            best = res
    return unpack(best.x)

def predict(row):
    return float(_simulate(row, fit([row]) if False else _MODEL_PARAMS))