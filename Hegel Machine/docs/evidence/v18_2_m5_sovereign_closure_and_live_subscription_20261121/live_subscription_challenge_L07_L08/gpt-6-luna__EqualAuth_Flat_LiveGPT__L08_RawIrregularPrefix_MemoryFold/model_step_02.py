import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    def z0(row, tau):
        if row.get('mem_conv') is not None:
            return float(row['mem_conv'])
        pref = row.get('raw_prefix') or []
        if not pref:
            return 0.52 * float(row['x_now'])
        ts = np.array([float(p['t']) for p in pref], dtype=float)
        xs = np.array([float(p['x']) for p in pref], dtype=float)
        z = 0.52 * xs[0]
        for i in range(1, len(ts)):
            dt = max(0.0, ts[i] - ts[i-1])
            avgx = 0.5 * (xs[i] + xs[i-1])
            e = math.exp(-dt / tau)
            z = e * z + 0.52 * avgx * (1.0 - e)
        return z

    def forecast(row, p):
        a, b, c, tau = p
        x = float(row['x_now'])
        z = z0(row, tau)
        u = float(row['u_now'])
        H = float(row['H'])
        n = max(1, int(math.ceil(H / 0.004)))
        h = H / n
        def rhs(xv, zv):
            return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
        for _ in range(n):
            k1x,k1z = rhs(x,z)
            k2x,k2z = rhs(x+0.5*h*k1x,z+0.5*h*k1z)
            k3x,k3z = rhs(x+0.5*h*k2x,z+0.5*h*k2z)
            k4x,k4z = rhs(x+h*k3x,z+h*k3z)
            x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
            z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
        return x

    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([forecast(r,p) for r in train_rows]) - y
    best = None
    for start in ([0.8,0.5,0.5,0.2], [1.2,0.8,0.7,0.5], [0.5,0.3,1.0,1.0]):
        res = least_squares(residual, np.array(start), bounds=([0.001,0.0,0.0,0.01],[5.0,5.0,5.0,5.0]), max_nfev=1500)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = _FIT_PARAMS
    a,b,c,tau = p
    if row.get('mem_conv') is not None:
        z = float(row['mem_conv'])
    else:
        pref = row.get('raw_prefix') or []
        if not pref:
            z = 0.52*float(row['x_now'])
        else:
            ts = [float(q['t']) for q in pref]; xs = [float(q['x']) for q in pref]
            z = 0.52*xs[0]
            for i in range(1,len(ts)):
                dt=max(0.0,ts[i]-ts[i-1]); e=math.exp(-dt/tau)
                z=e*z+0.52*0.5*(xs[i]+xs[i-1])*(1.0-e)
    x=float(row['x_now']); u=float(row['u_now']); H=float(row['H'])
    n=max(1,int(math.ceil(H/0.004))); h=H/n
    def rhs(xv,zv): return -a*xv+b*zv+c*u,(0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=rhs(x,z); k2x,k2z=rhs(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z=rhs(x+0.5*h*k2x,z+0.5*h*k2z); k4x,k4z=rhs(x+h*k3x,z+h*k3z)
        x+=h*(k1x+2*k2x+2*k3x+k4x)/6.0; z+=h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return float(x)

# The fit contract stores fitted parameters for prediction.
_FIT_PARAMS = np.array([1.0, 0.5, 0.5, 0.5])