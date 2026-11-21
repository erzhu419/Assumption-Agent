import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pts], dtype=float)
    xs = np.array([float(p['x']) for p in pts], dtype=float)
    s = ts - ts[0]
    z = 0.52 * xs[0]
    for i in range(len(s)-1):
        ds = s[i+1] - s[i]
        # Trapezoidal exponential-kernel integration on this irregular interval.
        z = z * math.exp(-ds/tau) + 0.52 * (xs[i] + xs[i+1]) * 0.5 * (1.0 - math.exp(-ds/tau))
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
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - y
    starts = ([0.5,0.3,0.5,0.5], [1.0,0.5,0.5,1.0], [0.2,0.1,0.8,2.0])
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([1e-6,1e-6,1e-6,1e-4],[10,10,10,20]), max_nfev=3000)
        if best is None or np.dot(res.fun,res.fun) < np.dot(best.fun,best.fun):
            best = res
    return best.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else row['params']
    return float(_simulate(row, p))