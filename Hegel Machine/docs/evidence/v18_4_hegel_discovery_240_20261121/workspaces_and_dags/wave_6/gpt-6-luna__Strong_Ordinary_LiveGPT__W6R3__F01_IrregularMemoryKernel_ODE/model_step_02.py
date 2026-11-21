import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _z0(row, tau):
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row.get('x_now', 0.0))
    ts = np.array([float(q['t']) for q in pref], dtype=float)
    xs = np.array([float(q['x']) for q in pref], dtype=float)
    order = np.argsort(ts)
    ts, xs = ts[order], xs[order]
    end = ts[-1]
    z = 0.52 * xs[0] * math.exp(-(end-ts[0])/tau)
    for i in range(len(ts)-1):
        d = ts[i+1]-ts[i]
        if d <= 0: continue
        # Integrate the linearly interpolated x forcing against the exponential kernel.
        q = d/tau
        e = math.exp(-q)
        w0 = (1.0-(1.0+q)*e)/q
        w1 = 1.0-w0
        z += 0.52 * (xs[i]*w0 + xs[i+1]*w1) * (1.0-e)
    return z

def _simulate(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H/0.004)))
    dt = H/n
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
    global _P
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r,p) for r in train_rows])-y
    starts = [[0.5,0.2,0.8,0.15],[1.0,0.5,1.0,0.3],[0.2,0.1,0.5,0.08]]
    best = None
    for s in starts:
        res = least_squares(residual, s, bounds=([0.001,-5,-5,0.005],[5,5,5,2]), max_nfev=3000)
        if best is None or np.dot(res.fun,res.fun)<np.dot(best.fun,best.fun): best=res
    _P = best.x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_simulate(row,_P))