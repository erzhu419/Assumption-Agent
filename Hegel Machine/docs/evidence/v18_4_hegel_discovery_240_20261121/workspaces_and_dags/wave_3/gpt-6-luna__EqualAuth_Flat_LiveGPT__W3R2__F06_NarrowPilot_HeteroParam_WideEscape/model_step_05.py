import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    global _p
    def f(p, r):
        j0, beta, rho, cs = p
        u=max(1e-8,float(r['u_now']))
        s=max(0.0,float(r['s_grad']))
        return j0*u*math.exp(-beta/(0.5+u))*(1-rho*s/(1+s))+cs*math.sqrt(u)
    y=np.array([r['y'] for r in train_rows],float)
    starts=[[5,1,.7,1],[10,2,.6,2],[3,.2,.8,1]]
    best=None
    for x0 in starts:
        res=least_squares(lambda p: np.array([f(p,r) for r in train_rows])-y,x0,bounds=([0,0,0,0],[100,20,1.5,30]),max_nfev=30000)
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    _p=best.x
    return _p

def predict(row):
    j0,beta,rho,cs=_p
    u=max(1e-8,float(row['u_now']))
    s=max(0.0,float(row['s_grad']))
    return float(j0*u*math.exp(-beta/(0.5+u))*(1-rho*s/(1+s))+cs*math.sqrt(u))