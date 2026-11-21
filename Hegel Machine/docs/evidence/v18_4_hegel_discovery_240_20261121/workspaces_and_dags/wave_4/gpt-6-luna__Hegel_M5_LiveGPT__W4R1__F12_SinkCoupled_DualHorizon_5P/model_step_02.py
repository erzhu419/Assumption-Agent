import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    def pred(p, r):
        vmax, rho, amp_adapt, tau_fast, tau_slow = p
        u = float(r['u_now']); s = float(r['s_grad']); H = float(r['H'])
        return vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u))*(-math.expm1(-H/tau_fast)) - amp_adapt*u*(-math.expm1(-H/tau_slow))
    def resid(p):
        return np.array([pred(p,r)-float(r['y']) for r in train_rows])
    starts = [[6.5,.7,.5,.4,15.], [7.,.7,.4,.5,20.], [5.,.65,.6,.3,12.]]
    best = None
    for x0 in starts:
        res = least_squares(resid, x0, bounds=([0.01,0.01,0.001,0.01,1.0],[30.0,.99,5.0,5.0,100.0]), max_nfev=5000)
        if best is None or np.dot(res.fun,res.fun)<np.dot(best.fun,best.fun): best=res
    return best.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else fit([])
    vmax,rho,amp_adapt,tau_fast,tau_slow = p
    u=float(row['u_now']); s=float(row['s_grad']); H=float(row['H'])
    return float(vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u))*(-math.expm1(-H/tau_fast))-amp_adapt*u*(-math.expm1(-H/tau_slow)))