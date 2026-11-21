import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z_from_prefix(row, tau):
    pref = row.get('raw_prefix')
    if pref:
        ts = np.asarray([p['t'] for p in pref], dtype=float)
        xs = np.asarray([p['x'] for p in pref], dtype=float)
        elapsed = ts[-1] - ts[0]
        conv = 0.0
        for i in range(1, len(ts)):
            d = ts[i] - ts[i-1]
            conv += 0.5 * (math.exp(-(ts[-1]-ts[i-1])/tau)*xs[i-1] + math.exp(-(ts[-1]-ts[i])/tau)*xs[i]) * d / tau
        return math.exp(-elapsed/tau) * 0.52 * xs[0] + 0.52 * conv
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    return 0.52 * float(row['x_now'])

def _integrate(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = _z_from_prefix(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def resid(p):
        return np.asarray([_integrate(r,p) for r in train_rows]) - y
    starts = [np.array([0.5,0.5,0.5,0.2]), np.array([1.0,1.0,0.2,0.5]), np.array([0.2,0.8,0.5,1.0])]
    best = None
    for x0 in starts:
        res = least_squares(resid, x0, bounds=([0.001,0.001,-5.0,0.005],[5.0,5.0,5.0,5.0]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_integrate(row, _PARAMS))