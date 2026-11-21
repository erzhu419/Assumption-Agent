import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    def predp(p, r):
        vmax, rho, c_loc, k_aval, s_crit = p
        u = r['u_now']
        s = r['s_grad']
        return vmax*(1-rho*s/(1+s))*(u/(1.2+u)) + c_loc*math.sqrt(max(u,0.0)) - k_aval*max(0.0,s-s_crit)*u
    def resid(p):
        return np.array([predp(p,r)-r['y'] for r in rows])
    starts = [np.array([6.5,0.70,0.1,2.0,2.58]), np.array([7.0,0.72,0.5,3.0,2.65]), np.array([6.0,0.68,-0.2,1.0,2.50])]
    best = None
    for x0 in starts:
        z = least_squares(resid, x0, bounds=([0.1,0.05,-3.0,0.0,2.36],[20.0,0.99,3.0,20.0,3.2]), max_nfev=30000)
        if best is None or np.sum(z.fun*z.fun)<np.sum(best.fun*best.fun): best=z
    return best.x

def predict(row):
    p = fit(_FIT_ROWS)
    vmax, rho, c_loc, k_aval, s_crit = p
    u=row['u_now']; s=row['s_grad']
    return float(vmax*(1-rho*s/(1+s))*(u/(1.2+u)) + c_loc*math.sqrt(max(u,0.0)) - k_aval*max(0.0,s-s_crit)*u)
