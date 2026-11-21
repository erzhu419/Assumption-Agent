import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    hist = row.get('raw_prefix') or []
    if not hist:
        return 0.52 * float(row['x_now'])
    ts = [float(p['t']) for p in hist]
    xs = [float(p['x']) for p in hist]
    z = 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        if dt > 0:
            avgx = 0.5 * (xs[i-1] + xs[i])
            e = math.exp(-dt / tau)
            z = z * e + 0.52 * avgx * (1.0 - e)
    return z

def _simulate(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z=f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = [[0.5,0.3,0.5,0.5],[1.0,0.5,1.0,1.0],[0.2,0.2,0.3,2.0]]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([1e-5,1e-5,1e-5,0.03],[10,10,10,10]), max_nfev=3000)
        if best is None or np.dot(result.fun,result.fun) < np.dot(best.fun,best.fun):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _PARAMS))