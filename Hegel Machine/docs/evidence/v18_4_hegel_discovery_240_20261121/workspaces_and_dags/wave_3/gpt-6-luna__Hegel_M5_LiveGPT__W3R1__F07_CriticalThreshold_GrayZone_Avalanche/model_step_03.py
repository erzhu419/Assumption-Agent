import math
import numpy as np
from scipy.optimize import least_squares

def _base(r, vmax, rho, c, k, s0):
    u = float(r['u_now'])
    s = float(r['s_grad'])
    return vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u)) + c*math.sqrt(max(u,0.0)) - k*max(0.0,s-s0)*u

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r['u_now']), float(r['s_grad'])] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        vmax,rho,c,k,s0 = p
        return np.array([_base(r,vmax,rho,c,k,s0) for r in rows])-y
    p0 = [6.5,0.70,0.1,1.0,2.59]
    lo = [1.0,0.05,-2.0,0.0,2.50]
    hi = [15.0,0.95,2.0,10.0,2.68]
    res = least_squares(residual,p0,bounds=(lo,hi),max_nfev=30000)
    return res.x

def predict(row):
    # Fit receives training rows when invoked by the evaluator; callers may pass fitted parameters instead.
    p = fit(_TRAIN_ROWS) if '_TRAIN_ROWS' in globals() else None
    if p is None:
        raise ValueError('fit parameters must be supplied through _TRAIN_ROWS before predict')
    return float(_base(row,*p))