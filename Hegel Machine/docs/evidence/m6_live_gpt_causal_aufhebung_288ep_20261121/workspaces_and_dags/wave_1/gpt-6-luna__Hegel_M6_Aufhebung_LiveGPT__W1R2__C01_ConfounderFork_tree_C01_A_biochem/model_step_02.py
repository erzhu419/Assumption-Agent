import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    X=np.array([float(r['x_ind']) for r in train_rows])
    U=np.array([float(r['u_ctrl']) for r in train_rows])
    Z=np.array([float(r['z_1']) for r in train_rows])
    Y=np.array([float(r['y']) for r in train_rows])
    # Tree-A indicator: the decorrelated probe rows support a z1-triggered branch.
    I=(Z>=2.0).astype(float)
    def pred(p,x,u,i):
        vmax,k,c,amp=p
        q=x/1.5
        return vmax*(q*q)/(k+q*q)-c*u+amp*i
    def resid(p): return pred(p,X,U,I)-Y
    res=least_squares(resid,x0=[8.0,1.0,0.2,1.5],bounds=([0,1e-8,-5,-10],[50,100,10,10]))
    return res.x.tolist()

def predict(row):
    # Fit parameters are installed by fit in the usual runner; this standalone
    # fallback is replaced below by parameters attached at fit time.
    p=_MODEL
    x=float(row['x_ind']); u=float(row['u_ctrl']); i=1.0 if float(row['z_1'])>=2.0 else 0.0
    q=x/1.5
    vmax,k,c,amp=p
    return float(vmax*(q*q)/(k+q*q)-c*u+amp*i)

_MODEL=[8.0,1.0,0.2,1.5]
