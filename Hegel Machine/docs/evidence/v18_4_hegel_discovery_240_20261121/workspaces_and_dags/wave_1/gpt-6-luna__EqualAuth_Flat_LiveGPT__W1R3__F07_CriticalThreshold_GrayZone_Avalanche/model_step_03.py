import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    return np.array([1.0, row['u_now'], row['s_grad'], row['u_now']*row['s_grad'], row['u_now']**2, row['s_grad']**2], dtype=float)

def fit(train_rows):
    X=np.vstack([_features(r) for r in train_rows])
    y=np.array([r['y'] for r in train_rows], dtype=float)
    # Fit a quadratic response surface; robust initialization and numerical refinement.
    b0=np.linalg.lstsq(X,y,rcond=None)[0]
    opt=least_squares(lambda b: X@b-y, b0, max_nfev=2000)
    return opt.x

def predict(row):
    b=_MODEL_PARAMS
    return float(_features(row)@b)
