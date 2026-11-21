import math
import numpy as np
from scipy.optimize import least_squares

_STATE = None

def fit(train_rows):
    global _STATE
    keys = ['x_ind', 'z_1', 'z_2', 'z_3', 'z_4']
    rows = list(train_rows)
    x = np.array([float(r['x_ind']) for r in rows])
    u = np.array([float(r.get('u_ctrl', 0.0)) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    candidates = []
    for key in keys:
        vals = sorted(set(float(r[key]) for r in rows))
        thresholds = [(a+b)/2 for a,b in zip(vals[:-1], vals[1:])] or [vals[0]]
        for threshold in thresholds:
            for direction in (1, -1):
                ind = np.array([1.0 if direction*(float(r[key])-threshold)>0 else 0.0 for r in rows])
                def residual(p):
                    vmax, k, c, d = p
                    h = x*x/(k+x*x)
                    return vmax*h - c*u + d*ind - y
                res = least_squares(residual, [7.0, 1.0, 0.1, 1.0], bounds=([0.0, 1e-6, -10.0, -10.0], [30.0, 100.0, 10.0, 10.0]))
                candidates.append((float(np.mean(res.fun**2)), key, threshold, direction, res.x))
    _, key, threshold, direction, p = min(candidates, key=lambda v: v[0])
    _STATE = (key, threshold, direction, p)
    return predict

def predict(row):
    key, threshold, direction, p = _STATE
    vmax, k, c, d = p
    x = float(row['x_ind'])
    u = float(row.get('u_ctrl', 0.0))
    ind = 1.0 if direction*(float(row[key])-threshold)>0 else 0.0
    return float(vmax*x*x/(k+x*x) - c*u + d*ind)
