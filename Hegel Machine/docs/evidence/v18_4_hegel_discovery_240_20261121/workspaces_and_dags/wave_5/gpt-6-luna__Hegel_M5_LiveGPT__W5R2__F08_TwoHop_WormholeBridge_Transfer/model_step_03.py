import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u=float(row['u_now']); s=float(row['s_grad']); H=float(row['H'])
    return np.array([u, u*u, u*u*u, s, s*u, s*u*u, H, H*u], dtype=float)

def fit(train_rows):
    X=np.vstack([_features(r) for r in train_rows])
    y=np.array([float(r['y']) for r in train_rows])
    # Ridge-stabilized initialization, followed by numerical least-squares fit.
    p0=np.linalg.lstsq(X.T@X+1e-8*np.eye(X.shape[1]), X.T@y, rcond=None)[0]
    res=least_squares(lambda p: X@p-y, p0, max_nfev=20000)
    return res.x

def predict(row):
    p=fit(_TRAIN_ROWS)
    return float(_features(row)@p)
