import numpy as np
from scipy.optimize import least_squares

def _features(r):
    u=float(r['u_now']); s=float(r['s_grad'])
    return np.array([1.0,u,u*u,s,s*s,u*s],dtype=float)

def fit(train_rows):
    X=np.vstack([_features(r) for r in train_rows])
    y=np.array([float(r['y']) for r in train_rows])
    scale=np.maximum(np.linalg.norm(X,axis=0),1.0)
    Z=X/scale
    res=least_squares(lambda c: Z@c-y, np.linalg.lstsq(Z,y,rcond=None)[0], max_nfev=5000)
    return res.x/scale

def predict(row):
    return float(_features(row) @ fit(_ROWS))