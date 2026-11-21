import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u=max(0.0,float(row['u_now']))
    s=float(row['s_grad'])
    return vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u)) + c_loc*math.sqrt(u) - k_aval*max(0.0,s-s_crit)*u

def fit(train_rows):
    ys=np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r,p) for r in train_rows])-ys
    starts=[[5,.5,.4,1,2.6],[4,.3,.2,2,2.5],[8,.7,.8,1.5,2.7]]
    sols=[least_squares(residual,x0,bounds=([0,0,0,0,2.0],[30,2,5,10,3.2]),max_nfev=30000) for x0 in starts]
    return min(sols,key=lambda z:np.sum(z.fun**2)).x

def predict(row):
    return float(_pred(row, fit(_TRAIN_ROWS)))