import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u)) + c_loc*math.sqrt(max(u,0.0)) - k_aval*max(0.0,s-s_crit)*u

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r,p) for r in train_rows])-ys
    starts = [[6.5,0.70,0.15,1.5,2.58],[6.5,0.70,0.15,2.5,2.65],[5.0,0.68,0.5,1.0,2.5]]
    best = None
    for x0 in starts:
        res = least_squares(residual,x0,bounds=([1.0,0.05,-2.0,0.0,2.36],[15.0,0.95,2.0,10.0,3.2]),max_nfev=20000)
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    return best.x

def predict(row):
    p = fit(train_rows)
    return float(_pred(row,p))