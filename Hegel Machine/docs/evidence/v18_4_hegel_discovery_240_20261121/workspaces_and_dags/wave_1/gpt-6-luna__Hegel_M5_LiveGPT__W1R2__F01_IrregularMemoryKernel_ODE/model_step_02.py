import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    def initial_z(row, tau):
        if row.get('mem_conv') is not None:
            return float(row['mem_conv'])
        pref = row.get('raw_prefix') or []
        if not pref:
            return 0.52 * float(row['x_now'])
        t0 = float(pref[0]['t'])
        tend = float(pref[-1]['t'])
        z = 0.52 * float(pref[0]['x']) * math.exp(-(tend-t0)/tau)
        for p, q in zip(pref[:-1], pref[1:]):
            a, b = float(p['t']), float(q['t'])
            xa, xb = float(p['x']), float(q['x'])
            da, db = tend-a, tend-b
            conv = 0.5*(math.exp(-da/tau)*xa + math.exp(-db/tau)*xb)*(b-a)/tau
            z += 0.52 * conv
        return z
    def simulate(row, pars):
        a, b, c, tau = pars
        x = float(row['x_now'])
        z = initial_z(row, tau)
        H = float(row['H'])
        n = max(1, int(math.ceil(H/0.004)))
        dt = H/n
        u = float(row['u_now'])
        def f(x,z):
            return (-a*x+b*z+c*u, (0.52*x-z)/tau)
        for _ in range(n):
            k1x,k1z=f(x,z)
            k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
            k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2)
            k4x,k4z=f(x+dt*k3x,z+dt*k3z)
            x += dt*(k1x+2*k2x+2*k3x+k4x)/6
            z += dt*(k1z+2*k2z+2*k3z+k4z)/6
        return x
    def residual(p):
        return np.array([simulate(r,p)-float(r['y']) for r in train_rows])
    starts = [[0.5,0.5,0.5,0.5],[1,1,1,1],[0.2,1,0.5,0.2],[1.5,0.2,1,1.5]]
    sols = [least_squares(residual,s,bounds=([0,0,0,0.03],[5,5,5,5]),max_nfev=2000) for s in starts]
    best=min(sols,key=lambda s: np.sum(s.fun**2))
    return {'params':best.x.tolist()}

def predict(row):
    p=row['_model']['params']
    a,b,c,tau=p
    if row.get('mem_conv') is not None:
        z=float(row['mem_conv'])
    else:
        pref=row.get('raw_prefix') or []
        if not pref: z=0.52*float(row['x_now'])
        else:
            end=float(pref[-1]['t']); start=float(pref[0]['t'])
            z=0.52*float(pref[0]['x'])*math.exp(-(end-start)/tau)
            for q,r in zip(pref[:-1],pref[1:]):
                t0,t1=float(q['t']),float(r['t']); x0,x1=float(q['x']),float(r['x'])
                z+=0.26*(math.exp(-(end-t0)/tau)*x0+math.exp(-(end-t1)/tau)*x1)*(t1-t0)/tau
    x=float(row['x_now']); H=float(row['H']); u=float(row['u_now'])
    n=max(1,int(math.ceil(H/0.004))); dt=H/n
    def f(x,z): return (-a*x+b*z+c*u,(0.52*x-z)/tau)
    for _ in range(n):
        k1=f(x,z); k2=f(x+dt*k1[0]/2,z+dt*k1[1]/2); k3=f(x+dt*k2[0]/2,z+dt*k2[1]/2); k4=f(x+dt*k3[0],z+dt*k3[1])
        x+=dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6; z+=dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
    return float(x)