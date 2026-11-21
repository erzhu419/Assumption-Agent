import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, c = p
        return vmax*x/(1.1+x) + c*u
    res = least_squares(lambda p: pred(p)-y, x0=np.array([5.0, 0.5]))
    return tuple(res.x)

def predict(row):
    vmax, c = fit([row])
    return float(vmax*float(row['x_ind'])/(1.1+float(row['x_ind'])) + c*float(row['u_ctrl']))