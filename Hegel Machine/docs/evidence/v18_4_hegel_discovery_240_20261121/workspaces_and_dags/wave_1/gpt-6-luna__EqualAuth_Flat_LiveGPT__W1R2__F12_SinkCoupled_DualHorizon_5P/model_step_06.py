import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a, b, c, tau = p
    H = max(0.0, float(row['H']))
    n = max(1, int(math.ceil(H / 0.15)))
    dt = H / n
    x = float(row['x_now'])
    z = float(row.get('mem_conv') or 0.0)
    u = float(row['u_now'])
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
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
    y=np.array([r['y'] for r in train_rows],float)
    def residual(p):
        return np.array([_simulate(r,p) for r in train_rows])-y
    best=None
    for start in ([.2,.2,1.,1.],[.1,.5,2.,2.],[.5,.05,3.,.5]):
        q=least_squares(residual,start,bounds=([0,0,0,.03],[3,3,10,30]),max_nfev=250)
        if best is None or np.dot(q.fun,q.fun)<np.dot(best.fun,best.fun): best=q
    _PARAMS=best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise ValueError('fit must be called first')
    return float(_simulate(row,_PARAMS))