import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    def pred(p, r):
        vmax, rho, c_loc, k_aval, s_crit = p
        u = max(0.0, r['u_now'])
        s = r['s_grad']
        return vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u)) + c_loc*math.sqrt(u) - k_aval*max(0.0,s-s_crit)*u
    def residual(p):
        return np.array([pred(p,r)-r['y'] for r in rows])
    best = None
    for start in ([5,0.4,0.2,2.0,2.6],[4,0.2,0.3,1.5,2.5],[6,0.5,0.1,3.0,2.7]):
        res = least_squares(residual, start, bounds=([0,0,0,0,2.0],[20,2,5,10,3.2]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2): best=res
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    vmax,rho,c_loc,k_aval,s_crit=p
    u=max(0.0,row['u_now']); s=row['s_grad']
    return float(vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u))+c_loc*math.sqrt(u)-k_aval*max(0.0,s-s_crit)*u)
