import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, r):
        a, b, c = p
        u = float(r['u_now'])
        g = float(r['s_grad'])
        return a*u/(1.0 + u/b) + c*(1.0-g)
    def residual(p):
        return np.array([pred(p,r)-float(r['y']) for r in rows])
    opt = least_squares(residual, x0=np.array([3.7,1.1,1.0]), bounds=([0.0,0.01,-10.0],[20.0,20.0,10.0]))
    return opt.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(p[0]*float(row['u_now'])/(1.0+float(row['u_now'])/p[1]) + p[2]*(1.0-float(row['s_grad'])))