import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    def pred(p, r):
        u=max(0.0,float(r['u_now']))
        s=float(r['s_grad'])
        H=float(r['H'])
        # Saturating substrate response with gradient-dependent inhibition; amplitude may vary smoothly with horizon.
        A, K, rho, c, h = p
        amp=A*(1.0+h*H)
        return amp*u/(K+u)*math.exp(-rho*s) + c*math.sqrt(u)
    def resid(p):
        return [pred(p,r)-float(r['y']) for r in rows]
    best=None
    for start in ([5,1,0.1,0.1,0],[3,0.5,0.5,0.2,0.1],[8,2,0.3,0.0,-0.1]):
        res=least_squares(resid,start,bounds=([0.01,0.01,0,0,-2],[30,20,5,5,2]),max_nfev=10000)
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    return best.x

def predict(row):
    p=fit(_TRAIN_ROWS)
    u=max(0.0,float(row['u_now'])); s=float(row['s_grad']); H=float(row['H'])
    A,K,rho,c,h=p
    return float(A*(1.0+h*H)*u/(K+u)*math.exp(-rho*s)+c*math.sqrt(u))