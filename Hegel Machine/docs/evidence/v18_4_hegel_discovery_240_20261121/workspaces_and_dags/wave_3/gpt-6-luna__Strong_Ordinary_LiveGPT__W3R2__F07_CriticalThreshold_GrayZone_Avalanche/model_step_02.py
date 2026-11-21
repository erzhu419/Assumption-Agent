import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    def pred(p, r):
        vmax, rho, c_loc, k_aval, s_crit = p
        u = float(r['u_now'])
        s = float(r['s_grad'])
        un = u
        return vmax*(1.0-rho*s/(1.0+s))*(un/(1.2+un)) + c_loc*math.sqrt(max(un,0.0)) - k_aval*max(0.0,s-s_crit)*un
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([pred(p,r) for r in rows])-y
    best = None
    for p0 in ([5,.7,0,1,2.6],[4,.5,0.1,2,2.55],[6,.8,-.1,.8,2.65]):
        res=least_squares(residual,p0,bounds=([0,0,-10,0,2.0],[20,1.5,10,10,3.0]),max_nfev=30000)
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    return best.x

def predict(row):
    p=fit(_TRAIN_ROWS)
    vmax,rho,c_loc,k_aval,s_crit=p
    u=float(row['u_now']); s=float(row['s_grad'])
    return float(vmax*(1-rho*s/(1+s))*(u/(1.2+u))+c_loc*math.sqrt(max(u,0))-k_aval*max(0,s-s_crit)*u)
