import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def fit(train_rows):
    global _P
    m = np.array([float(r['m_2']) for r in train_rows])
    u = np.array([float(r['u_raw']) for r in train_rows])
    t = np.array([float(r['t_proc']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, km, c_raw, beta_t = p
        return vmax * m / (km + m) + c_raw * np.sqrt(u) + beta_t * t
    res = least_squares(lambda p: pred(p) - y, x0=[8.0, 1.0, 1.0, 0.0], bounds=([0.0, 1e-8, -20.0, -20.0], [50.0, 50.0, 20.0, 20.0]))
    _P = res.x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, km, c_raw, beta_t = _P
    m2 = float(row['m_2'])
    u = float(row['u_raw'])
    t = float(row['t_proc'])
    return float(vmax * m2 / (km + m2) + c_raw * math.sqrt(u) + beta_t * t)
