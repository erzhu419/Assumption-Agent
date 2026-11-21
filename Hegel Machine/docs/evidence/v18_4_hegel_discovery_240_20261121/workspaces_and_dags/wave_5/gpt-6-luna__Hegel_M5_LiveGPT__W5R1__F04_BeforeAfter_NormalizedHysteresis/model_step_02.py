import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    d = row['u_now'] - row['u_past']
    x = row['x_now']
    # signed normalized step, exponential relaxation toward the snapshot state
    return d, x, row['H']

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, r):
        d, x, h = _features(r)
        a, b, tau = p
        return a*d*math.exp(-h/tau) + b*x
    def resid(p):
        return np.array([pred(p, r)-r['y'] for r in rows])
    result = least_squares(resid, [1.0, 1.0, 0.004], bounds=([-1000,-1000,1e-7],[1000,1000,10]))
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    d, x, h = _features(row)
    return float(p[0]*d*math.exp(-h/p[2]) + p[1]*x)
