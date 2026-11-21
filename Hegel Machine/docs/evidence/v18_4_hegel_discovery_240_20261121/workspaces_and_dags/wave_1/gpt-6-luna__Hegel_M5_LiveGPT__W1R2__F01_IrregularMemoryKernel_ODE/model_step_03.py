import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    pref=row.get('raw_prefix') or []
    if not pref:
        return float(row.get('mem_conv') or 0.0)
    ts=np.array([p['t'] for p in pref], dtype=float)
    xs=np.array([p['x'] for p in pref], dtype=float)
    d=ts[-1]-ts[0]
    z=0.52*xs[0]*math.exp(-d/tau)
    for i in range(1,len(pref)):
        dt=ts[i]-ts[i-1]
        q0=math.exp(-(ts[-1]-ts[i-1])/tau)
        q1=math.exp(-(ts[-1]-ts[i])/tau)
        z += 0.52*0.5*(q0*xs[i-1]+q1*xs[i])*dt/tau
    return z

def _integrate(row, p):
    a,b,c,tau=p
    n=max(1,int(math.ceil(float(row['H'])/0.004)))
    dt=float(row['H'])/n
    def f(x,z):
        return -a*x+b*z+c*float(row['u_now']), (0.52*x-z)/tau
    x=float(row['x_now']); z=_z0(row,tau)
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z=f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    rows=list(train_rows)
    y=np.array([r['y'] for r in rows],dtype=float)
    def residual(p):
        return np.array([_integrate(r,p) for r in rows])-y
    starts=[[0.5,0.5,0.5,0.5],[1.0,1.0,0.8,1.0],[0.2,0.3,1.0,2.0]]
    best=None
    for s in starts:
        res=least_squares(residual,s,bounds=([0.001,0.001,0.001,0.02],[5.0,5.0,5.0,10.0]),max_nfev=3000)
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    return {'params':best.x.tolist()}

def predict(row):
    model=row.get('_model')
    if model is None:
        raise ValueError('predict requires fitted parameters in row["_model"]')
    return float(_integrate(row,model['params']))