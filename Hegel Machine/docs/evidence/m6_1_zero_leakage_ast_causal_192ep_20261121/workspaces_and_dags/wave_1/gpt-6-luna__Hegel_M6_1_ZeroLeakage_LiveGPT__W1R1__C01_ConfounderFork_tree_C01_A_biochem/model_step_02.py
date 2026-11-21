import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    global _params
    X=np.array([float(r['x_ind']) for r in train_rows])
    U=np.array([float(r['u_ctrl']) for r in train_rows])
    Y=np.array([float(r['y']) for r in train_rows])
    # Saturating Hill flux with quadratic activation and linear control inhibition.
    def pred(p,x,u):
        vmax,k,c=p
        q=(x/1.5)**2
        return vmax*q/(k+q)-c*u
    best=None
    for start in ([8,1,0.4],[10,2,0.8],[6,0.5,0.1]):
        res=least_squares(lambda p: pred(p,X,U)-Y, start,
                          bounds=([0.01,0.001,-10],[100,100,10]))
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    _params=best.x
    return _params

def predict(row):
    vmax,k,c=_params
    x=float(row['x_ind']); u=float(row.get('u_ctrl',0.0))
    q=(x/1.5)**2
    return float(vmax*q/(k+q)-c*u)