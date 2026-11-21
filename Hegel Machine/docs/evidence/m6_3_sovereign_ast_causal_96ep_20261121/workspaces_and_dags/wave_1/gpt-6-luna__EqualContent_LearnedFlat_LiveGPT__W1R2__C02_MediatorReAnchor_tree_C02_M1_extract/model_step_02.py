import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_raw']) for r in train_rows])
    s = np.array([float(r['s_surf']) for r in train_rows])
    t = np.array([float(r['t_proc']) for r in train_rows])
    m1 = np.array([float(r['m_1']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, km, c_raw, d = p
        return vmax * s / (km + s) + c_raw * np.sqrt(np.maximum(u, 0.0)) - d * t * m1
    res = least_squares(lambda p: pred(p) - y, x0=[8.0, 1.0, 0.5, 1.0], bounds=([0.0, 1e-8, -10.0, -20.0], [30.0, 30.0, 10.0, 20.0]))
    _params = res.x
    return lambda row: predict(row)

def predict(row):
    vmax, km, c_raw, d = _params
    u = max(float(row['u_raw']), 0.0)
    s = float(row['s_surf'])
    t = float(row['t_proc'])
    m1 = float(row['m_1'])
    return float(vmax * s / (km + s) + c_raw * math.sqrt(u) - d * t * m1)
