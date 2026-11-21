import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([float(r['u_raw']) for r in train_rows])
    t = np.array([float(r['t_proc']) for r in train_rows])
    s = np.array([float(r['s_surf']) for r in train_rows])
    m1 = np.array([float(r['m_1']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, km, craw, gamma = p
        return vmax*s/(km+s) + craw*np.sqrt(np.maximum(u, 0.0)) - gamma*t*m1
    res = least_squares(lambda p: pred(p)-y, x0=[8.0, 1.0, 1.0, 0.8], bounds=([0.0, 1e-6, -10.0, -10.0], [30.0, 30.0, 10.0, 10.0]))
    _PARAMS = res.x
    return predict

def predict(row):
    vmax, km, craw, gamma = _PARAMS
    u = max(float(row['u_raw']), 0.0)
    s = float(row['s_surf'])
    t = float(row['t_proc'])
    m1 = float(row['m_1'])
    return float(vmax*s/(km+s) + craw*math.sqrt(u) - gamma*t*m1)
