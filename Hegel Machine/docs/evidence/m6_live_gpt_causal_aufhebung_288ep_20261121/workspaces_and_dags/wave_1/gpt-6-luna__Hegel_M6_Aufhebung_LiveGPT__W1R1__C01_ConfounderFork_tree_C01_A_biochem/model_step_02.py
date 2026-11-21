import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x=np.array([float(r['x_ind']) for r in train_rows])
    u=np.array([float(r['u_ctrl']) for r in train_rows])
    y=np.array([float(r['y']) for r in train_rows])
    def pred(p, xx, uu):
        vmax, k, c = p
        q=(xx/1.5)**2
        return vmax*q/(k+q)-c*uu
    p0=np.array([9.0, 1.0, 0.5])
    res=least_squares(lambda p: pred(p,x,u)-y, p0, bounds=([0.0,1e-8,0.0],[100.0,100.0,100.0]))
    return tuple(float(v) for v in res.x)

def predict(row):
    # Parameters are populated by fit via model parameters attached globally.
    p=_MODEL_PARAMS
    x=float(row['x_ind']); u=float(row['u_ctrl'])
    q=(x/1.5)**2
    return float(p[0]*q/(p[1]+q)-p[2]*u)

_MODEL_PARAMS=(9.0,1.0,0.5)
_ORIGINAL_FIT=fit
def fit(train_rows):
    global _MODEL_PARAMS
    _MODEL_PARAMS=_ORIGINAL_FIT(train_rows)
    return _MODEL_PARAMS
