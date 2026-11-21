import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    U=np.array([float(r['u_now']) for r in train_rows])
    S=np.array([float(r['s_grad']) for r in train_rows])
    Y=np.array([float(r['y']) for r in train_rows])
    def pred(p,u,s):
        a,b,c,d=p
        u=np.maximum(u,1e-8)
        return (a*u/(b+u)+c*np.sqrt(u))*np.exp(-d*(s-1.0))
    def resid(p):
        return pred(p,U,S)-Y
    starts=[[2,1,1,0.5],[3,2,0.5,0.3],[1,0.5,2,1]]
    fits=[least_squares(resid,x,bounds=([0,1e-5,0,-3],[20,20,20,3]),max_nfev=20000) for x in starts]
    p=min(fits,key=lambda z:np.sum(z.fun**2)).x
    return {'params':[float(x) for x in p]}

def predict(row):
    p=np.asarray(row.get('_model_params',[1,1,1,0]),dtype=float)
    u=max(float(row['u_now']),1e-8); s=float(row['s_grad'])
    a,b,c,d=p
    return float((a*u/(b+u)+c*math.sqrt(u))*math.exp(-d*(s-1.0)))