import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(pref[0]['x'])
    tprev = float(pref[0]['t'])
    for a, b in zip(pref[:-1], pref[1:]):
        dt = float(b['t']) - tprev
        xm = 0.5 * (float(a['x']) + float(b['x']))
        e = math.exp(-max(0.0, dt) / tau)
        z = 0.52 * xm + (z - 0.52 * xm) * e
        tprev = float(b['t'])
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
        return (-a*xv + b*zv + c*u, (0.52*xv - zv)/tau)
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z=f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    best = None
    for start in ([0.5,0.2,0.8,0.15], [1.0,0.5,1.0,0.3], [0.2,0.8,0.5,0.08]):
        result = least_squares(residual, start, bounds=([0.001,-5.0,-5.0,0.005],[5.0,5.0,5.0,3.0]), max_nfev=3000)
        if best is None or np.dot(result.fun,result.fun) < np.dot(best.fun,best.fun):
            best = result
    _params = best.x.copy()
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _params))