import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_raw']) for r in rows])
    s = np.array([float(r['m_2']) for r in rows])
    t = np.array([float(r['t_proc']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p, u, s, t):
        vmax, km, craw, ct = p
        return vmax*s/(km+s) + craw*np.sqrt(np.maximum(u, 0)) + ct*t
    res = least_squares(lambda p: pred(p,u,s,t)-y, x0=[8,1,1,-1], bounds=([0,1e-8,-10,-10],[30,30,10,10]))
    return {'p': res.x}

def predict(row):
    p = fit([])['p']
    return float(p[0]*float(row['m_2'])/(p[1]+float(row['m_2'])) + p[2]*math.sqrt(max(float(row['u_raw']),0)) + p[3]*float(row['t_proc']))