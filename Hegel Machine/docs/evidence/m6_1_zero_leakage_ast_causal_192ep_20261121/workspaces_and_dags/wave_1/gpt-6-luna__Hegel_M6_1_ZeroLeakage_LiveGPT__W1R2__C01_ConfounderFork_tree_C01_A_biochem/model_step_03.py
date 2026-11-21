import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x=np.array([float(r['x_ind']) for r in train_rows])
    z=np.array([float(r['z_1']) for r in train_rows])
    u=np.array([float(r['u_ctrl']) for r in train_rows])
    y=np.array([float(r['y']) for r in train_rows])
    # Fit Hill flux, additive control effect, and a thresholded tree-A contribution.
    def pred(p, x=x, z=z, u=u):
        vmax, kh, c, amp, threshold, sharp = p
        hill=vmax*(x/1.5)**2/(kh+(x/1.5)**2)
        indicator=1.0/(1.0+np.exp(np.clip(-sharp*(z-threshold),-60,60)))
        return hill-c*u+amp*indicator
    best=None
    for threshold0 in np.linspace(0.4,3.1,8):
        p0=[7.5,1.0,0.25,1.0,threshold0,4.0]
        res=least_squares(lambda p: pred(p)-y, p0,
             bounds=([0.01,0.01,-3,-8,0,0.1],[30,20,3,8,4,30]), max_nfev=4000)
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    return best.x

def predict(row):
    p=fit([row]) if False else None
    # Parameters are attached by fit via module-level state for the requested API.
    p=_MODEL
    x=float(row['x_ind']); z=float(row['z_1']); u=float(row['u_ctrl'])
    vmax, kh, c, amp, threshold, sharp=p
    return float(vmax*(x/1.5)**2/(kh+(x/1.5)**2)-c*u+amp/(1.0+math.exp(max(-60,min(60,-sharp*(z-threshold))))))

_MODEL=None
_old_fit=fit
def fit(train_rows):
    global _MODEL
    _MODEL=_old_fit(train_rows)
    return _MODEL