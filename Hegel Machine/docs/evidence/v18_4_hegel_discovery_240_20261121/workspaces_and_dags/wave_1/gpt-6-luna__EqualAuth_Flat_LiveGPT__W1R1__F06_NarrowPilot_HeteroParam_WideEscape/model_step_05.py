import numpy as np
from scipy.optimize import least_squares
import math

def fit(train_rows):
    rows = train_rows
    u = np.array([max(1e-8, float(r['u_now'])) for r in rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    # y = j0*u*exp(-beta/(0.5+u))*(1-rho*s/(1+s)) + c_sqrt*sqrt(u)
    def pred(p, r):
        j0, beta, rho, c = p
        u0 = max(1e-8, float(r['u_now']))
        s0 = max(0.0, float(r['s_grad']))
        return j0*u0*math.exp(-beta/(0.5+u0))*(1.0-rho*s0/(1.0+s0)) + c*math.sqrt(u0)
    def residual(p):
        return np.array([pred(p, r)-float(r['y']) for r in rows])
    best = None
    for p0 in ([3, 1, .7, 0], [5, 2, .5, -.2], [2, .2, .9, .2]):
        res = least_squares(residual, p0, bounds=([0, 0, 0, -10], [100, 20, 1.5, 10]), max_nfev=5000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    # The evaluator supplies the training set through fit; retain fitted parameters via closure-compatible row.
    p = row.get('_fitted_parameters')
    if p is None:
        raise ValueError('predict requires parameters returned by fit')
    j0, beta, rho, c = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return float(j0*u*math.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s)) + c*math.sqrt(u))