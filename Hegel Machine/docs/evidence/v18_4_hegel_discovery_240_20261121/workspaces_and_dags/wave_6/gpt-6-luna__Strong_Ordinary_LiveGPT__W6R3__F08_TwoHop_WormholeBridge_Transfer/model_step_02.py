import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    def pred(p, r):
        u=max(1e-9,float(r['u_now']))
        s=max(0.0,float(r['s_grad']))
        j0,beta,rho,c=p
        return j0*u*math.exp(-beta/(0.5+u))*(1-rho*s/(1+s))+c*math.sqrt(u)
    def residual(p):
        return np.array([pred(p,r)-float(r['y']) for r in train_rows])
    starts=[[3,1,.7,0],[4,.5,.5,0.2],[2,2,.8,-.2]]
    best=None
    for p0 in starts:
        res=least_squares(residual,p0,bounds=([0,0,0,-10],[20,20,2,10]),max_nfev=20000)
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    return best.x

def predict(row):
    p=fit(_TRAIN_ROWS)
    u=max(1e-9,float(row['u_now'])); s=max(0.0,float(row['s_grad']))
    j0,beta,rho,c=p
    return float(j0*u*math.exp(-beta/(0.5+u))*(1-rho*s/(1+s))+c*math.sqrt(u))