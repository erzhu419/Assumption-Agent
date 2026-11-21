import numpy as np
from scipy.optimize import least_squares

def _features(r):
    u=float(r['u_now']); x=float(r['x_now']); m=float(r['mem_conv']); g=float(r['s_grad']); h=float(r['H'])
    q=g/(1.0+g)
    return np.array([1.0,u,x,m,q,h,u*q,u*h,x*h,m*h],dtype=float)

def fit(train_rows):
    X=np.vstack([_features(r) for r in train_rows]); y=np.array([float(r['y']) for r in train_rows])
    def residual(p): return X@p-y
    p0=np.linalg.lstsq(X,y,rcond=None)[0]
    result=least_squares(residual,p0,max_nfev=5000)
    global _MODEL
    _MODEL={'params':result.x}
    return _MODEL

def predict(row):
    return float(_features(row)@_MODEL['params'])