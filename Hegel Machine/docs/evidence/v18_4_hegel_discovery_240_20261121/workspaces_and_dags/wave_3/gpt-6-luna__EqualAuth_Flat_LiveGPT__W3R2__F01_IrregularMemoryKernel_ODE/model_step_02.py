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
    z = 0.52 * float(hist[0]['x'])
    t0 = float(hist[0]['t'])
    for k in range(1, len(hist)):
        t1 = float(hist[k]['t'])
        x0, x1 = float(hist[k-1]['x']), float(hist[k]['x'])
        n = max(1, int(math.ceil((t1-t0)/0.004)))
        h = (t1-t0)/n
        for j in range(n):
            xa = x0 + (x1-x0)*j/n
            xb = x0 + (x1-x0)*(j+1)/n
            xm = 0.5*(xa+xb)
            # RK4 for dz/dt=(0.52*x-z)/tau with linearly interpolated x
            frac = (j+0.5)/n
            def f(zz, ff):
                xx = x0 + (x1-x0)*ff
                return (0.52*xx-zz)/tau
            f1=f(z,j/n); f2=f(z+0.5*h*f1,frac); f3=f(z+0.5*h*f2,frac); f4=f(z+h*f3,(j+1)/n)
            z += h*(f1+2*f2+2*f3+f4)/6.0
        t0=t1
    return z

def _simulate(row, p):
    a,b,c,tau = p
    x=float(row['x_now']); z=_z0(row,tau); u=float(row['u_now'])
    H=float(row['H']); n=max(1,int(math.ceil(H/0.004))); h=H/n
    def f(xx,zz): return (-a*xx+b*zz+c*u, (0.52*xx-zz)/tau)
    for _ in range(n):
        k1=f(x,z); k2=f(x+h*k1[0]/2,z+h*k1[1]/2); k3=f(x+h*k2[0]/2,z+h*k2[1]/2); k4=f(x+h*k3[0],z+h*k3[1])
        x += h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
        z += h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
    return x

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_simulate(r,p)-float(r['y']) for r in train_rows])
    best=None
    for guess in ([0.4,0.4,0.4,0.5],[0.8,0.5,0.5,1.0],[0.2,0.2,0.8,1.5]):
        res=least_squares(residual,guess,bounds=([0,0,0,0.03],[5,5,5,10]),max_nfev=300)
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    _PARAMS=best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row,_PARAMS))