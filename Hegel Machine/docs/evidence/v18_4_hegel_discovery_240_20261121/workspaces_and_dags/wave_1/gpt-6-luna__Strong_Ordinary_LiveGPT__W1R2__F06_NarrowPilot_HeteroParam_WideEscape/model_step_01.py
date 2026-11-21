import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    inhib = 1.0 - row_rho_s(s) # placeholder overwritten in model
    return u, s

def row_rho_s(s, rho=0.0):
    return rho * s / (1.0 + s)

def fit(train_rows):
    rows = list(train_rows)
    def pred_params(p, r):
        j0, beta, rho, c = p
        u = max(1e-9, float(r['u_now']))
        s = float(r['s_grad'])
        return j0*u*math.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s)) + c*math.sqrt(u)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([pred_params(p,r) for r in rows])-y
    best = None
    for start in ([2.0,0.5,0.3,0.0],[1.0,1.0,0.7,0.5],[3.0,0.1,0.0,-0.5]):
        res = least_squares(residual, start, bounds=([0,0,0,-10],[100,20,1,10]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    p = best.x
    def predict(row):
        return float(pred_params(p,row))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first to obtain predict(row).')