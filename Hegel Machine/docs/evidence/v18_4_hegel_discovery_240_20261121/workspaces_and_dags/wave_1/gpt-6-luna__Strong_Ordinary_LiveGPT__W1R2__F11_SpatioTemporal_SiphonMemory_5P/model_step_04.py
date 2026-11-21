import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.002)))
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
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r,p) for r in train_rows])-y
    best = None
    for start in ([0.5,0.1,0.5,0.5],[1.0,0.5,1.0,1.0],[0.2,0.8,0.2,0.2],[1.5,0.05,1.5,0.8]):
        res = least_squares(residual, start, bounds=([0,0,0,0.01],[10,10,10,10]), max_nfev=5000)
        if best is None or np.dot(res.fun,res.fun)<np.dot(best.fun,best.fun): best=res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise ValueError('Call fit(train_rows) first')
    return float(_simulate(row,_PARAMS))