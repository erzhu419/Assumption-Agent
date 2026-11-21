import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, r):
        j0, beta, rho, c = p
        u = max(1e-8, float(r['u_now']))
        s = float(r['s_grad'])
        return j0*u*math.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s)) + c*math.sqrt(u)
    def residual(p):
        return np.array([pred(p,r)-float(r['y']) for r in rows])
    best = None
    for rho0 in (0.70, 0.5, 0.9):
        res = least_squares(residual, [3.0, 1.0, rho0, 0.0], bounds=([0.0,0.0,0.0,-5.0],[30.0,20.0,1.5,5.0]), max_nfev=20000)
        if best is None or np.dot(res.fun,res.fun)<np.dot(best.fun,best.fun): best=res
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    u = max(1e-8,float(row['u_now']))
    s = float(row['s_grad'])
    j0,beta,rho,c = p
    return float(j0*u*math.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s)) + c*math.sqrt(u))
