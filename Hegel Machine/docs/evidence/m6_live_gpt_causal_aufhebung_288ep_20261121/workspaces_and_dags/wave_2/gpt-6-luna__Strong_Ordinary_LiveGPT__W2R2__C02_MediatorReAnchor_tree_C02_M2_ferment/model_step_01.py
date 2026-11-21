import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    s=np.array([r['s_surf'] for r in train_rows],float)
    u=np.array([r['u_raw'] for r in train_rows],float)
    y=np.array([r['y'] for r in train_rows],float)
    def pred(p,s,u):
        vmax,km,c=p
        return vmax*s/(km+s)+c*np.sqrt(np.maximum(u,0.0))
    res=least_squares(lambda p: pred(p,s,u)-y,x0=[8.0,1.5,0.5],bounds=([0.0,1e-8,-20.0],[100.0,100.0,20.0]))
    fit.params=res.x
    return fit

def predict(row):
    vmax,km,c=fit.params
    s=float(row['s_surf']); u=max(0.0,float(row['u_raw']))
    return float(vmax*s/(km+s)+c*np.sqrt(u))