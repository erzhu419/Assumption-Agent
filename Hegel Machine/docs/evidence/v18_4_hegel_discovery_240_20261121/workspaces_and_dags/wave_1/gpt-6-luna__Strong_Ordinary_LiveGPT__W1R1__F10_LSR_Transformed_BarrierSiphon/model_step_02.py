import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u=np.array([r['u_now'] for r in train_rows],float)
    s=np.array([r['s_grad'] for r in train_rows],float)
    y=np.array([r['y'] for r in train_rows],float)
    def pred(p):
        j,b,rho,c=p
        return j*u*np.exp(-b/(0.5+u))*(1-rho*s/(1+s))+c*np.sqrt(u)
    best=None
    for p0 in ([8,0.5,0.7,0.2],[10,1,0.5,0],[6,0.2,0.8,0.5],[12,2,0.3,-0.2]):
        res=least_squares(lambda p: pred(p)-y,p0,bounds=([0,0,-3,-10],[100,20,3,10]),max_nfev=20000)
        loss=float(np.sum(res.fun**2))
        if best is None or loss<best[0]: best=(loss,res.x)
    return tuple(float(v) for v in best[1])

def predict(row):
    j,b,rho,c=fit(_TRAIN_ROWS)
    u=float(row['u_now']); s=float(row['s_grad'])
    return float(j*u*math.exp(-b/(0.5+u))*(1-rho*s/(1+s))+c*math.sqrt(u))
