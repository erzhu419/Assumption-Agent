import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    scale = 0.004
    def pred(p, r):
        gain, tau, keq = p
        un = r['u_now'] / scale
        up = r['u_past'] / scale
        return gain * (un - up) * math.exp(-r['H'] / tau) + keq * un
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def resid(p):
        return np.array([pred(p, r) for r in train_rows]) - y
    res = least_squares(resid, x0=[1.0, 0.5, 0.0], bounds=([-100, 1e-6, -100], [100, 100, 100]))
    return res.x.tolist()

def predict(row):
    p = fit([row])
    return p
