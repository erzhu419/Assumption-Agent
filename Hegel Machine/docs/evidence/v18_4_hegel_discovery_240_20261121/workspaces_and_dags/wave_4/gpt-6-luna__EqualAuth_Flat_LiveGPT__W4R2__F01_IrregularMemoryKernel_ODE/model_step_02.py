import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pts], dtype=float)
    xs = np.array([float(p['x']) for p in pts], dtype=float)
    t_end = ts[-1]
    z = 0.52 * xs[0] * math.exp(-(t_end-ts[0])/tau)
    for i in range(len(ts)-1):
        dt = ts[i+1]-ts[i]
        q0 = 0.52*xs[i]*math.exp(-(t_end-ts[i])/tau)/tau
        q1 = 0.52*xs[i+1]*math.exp(-(t_end-ts[i+1])/tau)/tau
        z += 0.5*dt*(q0+q1)
    return z

def _forecast(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H/0.004)))
    dt = H/n
    def f(xv, zv):
        return (-a*xv+b*zv+c*u, (0.52*xv-zv)/tau)
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
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_forecast(r,p) for r in train_rows])-ys
    starts = [[0.4,0.5,0.5,0.3],[0.8,0.8,0.8,0.8],[1.2,0.3,0.7,1.5]]
    sols = [least_squares(residual,s,bounds=([0.001,0.001,0.001,0.03],[5,5,5,5]),max_nfev=3000) for s in starts]
    _P = min(sols,key=lambda q: np.sum(q.fun*q.fun)).x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('call fit(train_rows) first')
    return float(_forecast(row,_P))