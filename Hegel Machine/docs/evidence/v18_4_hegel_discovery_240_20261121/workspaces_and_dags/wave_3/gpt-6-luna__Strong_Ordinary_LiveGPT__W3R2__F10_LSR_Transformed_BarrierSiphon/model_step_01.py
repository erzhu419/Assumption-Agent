import numpy as np
from scipy.optimize import least_squares
import math

def fit(train_rows):
    tau=3.0
    def pred(p,r):
        j0,beta,rho,c=p
        u=float(r['u_now'])/tau
        s=float(r['s_grad'])
        return j0*u*math.exp(-beta/(0.5+u))*(1-rho*s/(1+s))+c*math.sqrt(max(u,0.0))
    def residual(p):
        return np.array([pred(p,r)-float(r['y']) for r in train_rows])
    p0=np.array([4.0,1.0,0.7,0.1])
    res=least_squares(residual,p0,bounds=([0,0,0,-np.inf],[np.inf,np.inf,1,np.inf]),max_nfev=20000)
    return res.x

def predict(row):
    j0,beta,rho,c=fit(_fit_rows)
    u=float(row['u_now'])/3.0
    s=float(row['s_grad'])
    return float(j0*u*math.exp(-beta/(0.5+u))*(1-rho*s/(1+s))+c*math.sqrt(max(u,0.0)))