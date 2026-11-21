import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([float(r['u_raw']) for r in train_rows])
    t = np.array([float(r['t_proc']) for r in train_rows])
    s = np.array([float(r['s_surf']) for r in train_rows])
    m = np.array([[float(r['m_1']), float(r['m_2']), float(r['m_3'])] for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, km, craw, b1, b2, b3, b0 = p
        return vmax*s/(km+s) + craw*np.sqrt(np.maximum(u, 0.0)) + t*(b0 + m[:, 0]*b1 + m[:, 1]*b2 + m[:, 2]*b3)
    best = None
    for x0 in ([8, 1, 1, 0, 0, 0, 0], [10, 2, 0.5, -0.2, 0.2, 0.1, 0]):
        res = least_squares(lambda p: pred(p)-y, x0=x0, bounds=([0, 1e-6, -10, -10, -10, -10, -10], [30, 30, 10, 10, 10, 10, 10]), max_nfev=10000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return predict

def predict(row):
    vmax, km, craw, b1, b2, b3, b0 = _PARAMS
    u = max(float(row['u_raw']), 0.0)
    s = float(row['s_surf'])
    t = float(row['t_proc'])
    m1, m2, m3 = float(row['m_1']), float(row['m_2']), float(row['m_3'])
    return float(vmax*s/(km+s) + craw*math.sqrt(u) + t*(b0 + b1*m1 + b2*m2 + b3*m3))