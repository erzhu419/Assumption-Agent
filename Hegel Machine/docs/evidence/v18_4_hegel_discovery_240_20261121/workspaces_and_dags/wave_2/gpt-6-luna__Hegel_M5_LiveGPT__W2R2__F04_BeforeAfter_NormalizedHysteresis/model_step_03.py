import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Fit normalized before/after contrast: gain*(u_now-u_past)/tau_scale
    # times exponential relaxation, plus equilibrium response k_eq*u_now/tau_scale.
    tau_scale = 5.0
    def pred(p, row):
        gain, tau_relax, k_eq = p
        un = row['u_now'] / tau_scale
        up = row['u_past'] / tau_scale
        return gain * (un-up) * math.exp(-row['H']/tau_relax) + k_eq * un
    def residual(p):
        return np.array([pred(p,r)-r['y'] for r in train_rows])
    starts = ([1.0,5.0,0.0], [2.0,2.0,0.5], [0.5,10.0,-0.5])
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([-20.0,0.05,-20.0],[20.0,100.0,20.0]))
        if best is None or np.dot(res.fun,res.fun) < np.dot(best.fun,best.fun):
            best = res
    return best.x

def predict(row):
    p = fit(train_rows)
    un = row['u_now'] / 5.0
    up = row['u_past'] / 5.0
    return float(p[0] * (un-up) * math.exp(-row['H']/p[1]) + p[2] * un)
