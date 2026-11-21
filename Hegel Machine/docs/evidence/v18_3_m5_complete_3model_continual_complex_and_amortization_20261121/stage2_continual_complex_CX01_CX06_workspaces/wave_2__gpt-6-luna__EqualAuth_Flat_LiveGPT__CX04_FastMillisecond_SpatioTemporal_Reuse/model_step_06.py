import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _sim(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else 0.0
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
    global _P
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_sim(r,p) for r in train_rows])-y
    best = None
    for start in ([0.5,0.1,1.0,0.01],[1.0,0.5,2.0,0.005],[0.1,0.0,0.5,0.02]):
        q = least_squares(residual, start, bounds=([0.0,-5.0,-20.0,0.0002],[20.0,5.0,20.0,2.0]), max_nfev=400)
        if best is None or np.dot(q.fun,q.fun)<np.dot(best.fun,best.fun): best=q
    _P = best.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_sim(row,_P))