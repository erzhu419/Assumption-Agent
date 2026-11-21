import math
import numpy as np
from scipy.optimize import least_squares

def _features(r):
    o=float(r['u_outdoor']); i=float(r['u_indoor']); w=float(r['p_weather'])
    a=float(r['a_foot']); q=float(r['impact_idx'])
    return np.array([1.0, o*w, i, i*i, a, q, a*q, i*q, o*w*q], dtype=float)

def fit(train_rows):
    X=np.vstack([_features(r) for r in train_rows])
    y=np.array([float(r['y']) for r in train_rows])
    # Mild ridge regularization stabilizes this small sample without forcing
    # the intercept or primary indoor/outdoor effects toward zero.
    penalty=np.diag([0.0,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03])
    def residual(b):
        return np.r_[X@b-y, np.sqrt(penalty)@b]
    return least_squares(residual, np.zeros(X.shape[1])).x.tolist()

def predict(row):
    b=_MODEL
    return float(_features(row)@np.asarray(b))

_MODEL=None
_ORIGINAL_FIT=fit
def fit(train_rows):
    global _MODEL
    _MODEL=_ORIGINAL_FIT(train_rows)
    return _MODEL