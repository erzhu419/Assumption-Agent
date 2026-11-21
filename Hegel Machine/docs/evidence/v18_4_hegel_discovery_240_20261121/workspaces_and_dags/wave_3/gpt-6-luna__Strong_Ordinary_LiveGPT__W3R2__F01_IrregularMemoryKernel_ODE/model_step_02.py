import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    prefix = row.get('raw_prefix') or []
    if len(prefix) < 2:
        return 0.52 * float(row['x_now'])
    ts = np.asarray([float(p['t']) for p in prefix], dtype=float)
    xs = np.asarray([float(p['x']) for p in prefix], dtype=float)
    ts = ts - ts[0]
    z = 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        target0, target1 = 0.52 * xs[i-1], 0.52 * xs[i]
        e = math.exp(-dt / tau)
        slope = (target1 - target0) / dt if dt > 0 else 0.0
        z = target1 - slope * tau + (z - target0 + slope * tau) * e
    return z

def _simulate(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    h = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row, tau)
    u = float(row['u_now'])
    n = max(1, int(math.ceil(h / 0.004)))
    dt = h / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv - zv)/tau
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
    def residual(p):
        return np.asarray([_simulate(r,p)-float(r['y']) for r in train_rows])
    best = None
    for start in ([0.4,0.2,0.3,0.12], [0.8,0.1,0.5,0.2], [0.2,0.4,0.1,0.08]):
        result = least_squares(residual, start, bounds=([0.001,0.0,0.0,0.005],[5.0,5.0,5.0,2.0]), max_nfev=3000)
        if best is None or np.dot(result.fun,result.fun) < np.dot(best.fun,best.fun):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_simulate(row, _PARAMS))