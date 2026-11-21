import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pts])
    xs = np.array([float(p['x']) for p in pts])
    T = ts[-1]
    z = 0.52 * xs[0] * math.exp(-(T-ts[0])/tau)
    for i in range(len(ts)-1):
        dt = ts[i+1]-ts[i]
        n = max(1, int(math.ceil(dt/0.004)))
        h = dt/n
        xleft, xright = xs[i], xs[i+1]
        for j in range(n):
            xa = xleft+(xright-xleft)*j/n
            xb = xleft+(xright-xleft)*(j+1)/n
            z = z*math.exp(-h/tau) + 0.52*(xa+xb)*0.5*(1-math.exp(-h/tau))
    return z

def _integrate(row, pars):
    a,b,c,tau = pars
    x=float(row['x_now']); z=_z0(row,tau); u=float(row['u_now'])
    H=float(row['H']); n=max(1,int(math.ceil(H/0.004))); h=H/n
    def f(x,z): return -a*x+b*z+c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x,k1z=f(x,z); k2x,k2z=f(x+h*k1x/2,z+h*k1z/2)
        k3x,k3z=f(x+h*k2x/2,z+h*k2z/2); k4x,k4z=f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6
        z += h*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    global _PARAMS
    y=np.array([float(r['y']) for r in train_rows])
    def residual(q):
        pars=np.exp(q)
        return np.array([_integrate(r,pars) for r in train_rows])-y
    starts=[np.log([0.3,0.2,0.5,0.2]),np.log([0.7,0.4,0.8,0.5]),np.log([1.2,0.1,1.0,1.0])]
    best=None
    for q0 in starts:
        res=least_squares(residual,q0,bounds=(np.log([0.001,0.001,0.001,0.01]),np.log([10,10,10,10])),max_nfev=3000)
        if best is None or np.dot(res.fun,res.fun)<np.dot(best.fun,best.fun): best=res
    _PARAMS=np.exp(best.x)
    return {'a_slow':float(_PARAMS[0]),'b_couple':float(_PARAMS[1]),'c_drive':float(_PARAMS[2]),'tau_fast':float(_PARAMS[3])}

def predict(row):
    global _PARAMS
    if row.get('params') is not None:
        p=row['params']; pars=[p[k] for k in ('a_slow','b_couple','c_drive','tau_fast')]
    elif _PARAMS is not None:
        pars=_PARAMS
    else:
        raise ValueError('fit(train_rows) must be called before predict(row)')
    return float(_integrate(row,pars))