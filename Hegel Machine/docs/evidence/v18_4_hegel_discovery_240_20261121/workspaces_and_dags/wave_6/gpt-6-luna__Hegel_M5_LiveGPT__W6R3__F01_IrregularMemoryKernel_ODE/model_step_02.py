import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def predict_with(p, row):
        a, b, c, tau = [float(v) for v in p]
        pref = row.get('raw_prefix')
        if row.get('mem_conv') is not None:
            z0 = float(row['mem_conv'])
        elif pref:
            ts = np.array([float(q['t']) for q in pref], dtype=float)
            xs = np.array([float(q['x']) for q in pref], dtype=float)
            te = ts[-1]
            dt = np.diff(ts)
            conv = 0.0
            for k in range(1, len(ts)):
                conv += 0.5 * (math.exp(-(te-ts[k-1])/tau)*xs[k-1] + math.exp(-(te-ts[k])/tau)*xs[k]) * dt[k-1] / tau
            z0 = math.exp(-(te-ts[0])/tau) * 0.52 * xs[0] + 0.52 * conv
        else:
            z0 = 0.52 * float(row['x_now'])
        x = float(row['x_now'])
        z = z0
        u = float(row['u_now'])
        H = float(row['H'])
        n = max(1, int(math.ceil(H / 0.004)))
        h = H / n
        def f(xx, zz):
            return -a*xx + b*zz + c*u, (0.52*xx-zz)/tau
        for _ in range(n):
            k1x,k1z=f(x,z)
            k2x,k2z=f(x+h*k1x/2,z+h*k1z/2)
            k3x,k3z=f(x+h*k2x/2,z+h*k2z/2)
            k4x,k4z=f(x+h*k3x,z+h*k3z)
            x += h*(k1x+2*k2x+2*k3x+k4x)/6
            z += h*(k1z+2*k2z+2*k3z+k4z)/6
        return x
    def residual(p):
        return np.array([predict_with(p,r)-float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([0.5,0.2,0.5,0.2]), bounds=([0.001,0.0,-5.0,0.005],[5.0,5.0,5.0,5.0]), max_nfev=3000)
    return {'params': result.x.tolist()}

def predict(row):
    p = fit([row]).get('params')
    return float('nan') if p is None else float(_predict_saved(p,row))

def _predict_saved(p,row):
    a,b,c,tau = [float(v) for v in p]
    pref=row.get('raw_prefix')
    if row.get('mem_conv') is not None: z=float(row['mem_conv'])
    elif pref:
        ts=[float(q['t']) for q in pref]; xs=[float(q['x']) for q in pref]; te=ts[-1]
        conv=sum(0.5*(math.exp(-(te-ts[k-1])/tau)*xs[k-1]+math.exp(-(te-ts[k])/tau)*xs[k])*(ts[k]-ts[k-1])/tau for k in range(1,len(ts)))
        z=math.exp(-(te-ts[0])/tau)*0.52*xs[0]+0.52*conv
    else: z=0.52*float(row['x_now'])
    x=float(row['x_now']); u=float(row['u_now']); H=float(row['H']); n=max(1,int(math.ceil(H/0.004))); h=H/n
    def f(xx,zz): return -a*xx+b*zz+c*u,(0.52*xx-zz)/tau
    for _ in range(n):
        k1x,k1z=f(x,z); k2x,k2z=f(x+h*k1x/2,z+h*k1z/2); k3x,k3z=f(x+h*k2x/2,z+h*k2z/2); k4x,k4z=f(x+h*k3x,z+h*k3z)
        x+=h*(k1x+2*k2x+2*k3x+k4x)/6; z+=h*(k1z+2*k2z+2*k3z+k4z)/6
    return x