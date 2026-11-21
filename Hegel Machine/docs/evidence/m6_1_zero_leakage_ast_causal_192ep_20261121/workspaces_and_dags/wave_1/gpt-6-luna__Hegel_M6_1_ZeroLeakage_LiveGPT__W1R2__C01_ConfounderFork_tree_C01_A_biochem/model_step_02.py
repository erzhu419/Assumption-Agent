import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    x = np.array([float(r['x_ind']) for r in rows])
    z = np.array([float(r['z_1']) for r in rows])
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    vals = sorted(set(z.tolist()))
    thresholds = [-1e9] + [(a+b)/2 for a,b in zip(vals[:-1], vals[1:])] + [1e9]
    best = None
    for t in thresholds:
        ind = (z > t).astype(float)
        def resid(p):
            vmax, k, c, amp = p
            q = (x/1.5)**2
            return vmax*q/(k+q) - c*u + amp*ind - y
        res = least_squares(resid, [8.0, 1.0, 0.5, 1.5], bounds=([0.01,0.001,-10,-20],[30,100,10,20]))
        err = float(np.mean(res.fun**2))
        if best is None or err < best[0]:
            best = (err, t, res.x)
    return {'threshold': best[1], 'params': best[2].tolist()}

def predict(row):
    t = float(model['threshold'])
    vmax, k, c, amp = model['params']
    q = (float(row['x_ind'])/1.5)**2
    return float(vmax*q/(k+q) - c*float(row['u_ctrl']) + amp*(float(row['z_1']) > t))

model = None
_old_fit = fit
def fit(train_rows):
    global model
    model = _old_fit(train_rows)
    return model
