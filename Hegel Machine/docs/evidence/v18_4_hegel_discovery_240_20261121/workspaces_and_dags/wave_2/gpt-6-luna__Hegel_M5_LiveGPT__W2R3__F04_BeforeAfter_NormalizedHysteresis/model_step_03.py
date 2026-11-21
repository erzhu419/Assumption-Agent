import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    up=float(row['u_past']); un=float(row['u_now']); H=float(row['H'])
    denom=up+un
    z=(un-up)/denom if denom else 0.0
    return np.array([z, z*H, z*H*H, z*H*H*H], dtype=float)

def fit(train_rows):
    X=np.vstack([_features(r) for r in train_rows])
    y=np.array([float(r['y'])-float(r['x_now']) for r in train_rows])
    res=least_squares(lambda p: X@p-y, np.zeros(4))
    return res.x

def predict(row):
    p=_fit_parameters
    return float(row['x_now'])+float(_features(row)@p)
