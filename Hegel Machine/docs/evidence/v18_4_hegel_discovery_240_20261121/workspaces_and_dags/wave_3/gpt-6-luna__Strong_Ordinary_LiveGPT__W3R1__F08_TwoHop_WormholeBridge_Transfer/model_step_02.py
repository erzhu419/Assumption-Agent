import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u=np.array([r['u_now'] for r in train_rows],float)
    s=np.array([r['s_grad'] for r in train_rows],float)
    y=np.array([r['y'] for r in train_rows],float)
    def pred(p,u,s):
        vmax,rho,c=p
        un=u
        return vmax*(1-rho*s/(1+s))*(un/(1.2+un)) + c*np.sqrt(np.maximum(un,0))
    best=None
    for sign in (-1,1):
        p0=[6.0,0.5,sign*0.5]
        res=least_squares(lambda p: pred(p,u,s)-y,p0,bounds=([0,0,-10],[30,2,10]))
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    return best.x

def predict(row):
    p=fit([row])
    u=float(row['u_now']); s=float(row['s_grad'])
    vmax,rho,c=p
    return float(vmax*(1-rho*s/(1+s))*(u/(1.2+u))+c*np.sqrt(max(u,0)))