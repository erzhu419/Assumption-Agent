import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    m = np.array([float(r['m_2']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_raw']) for r in train_rows], dtype=float)
    t = np.array([float(r['t_proc']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(p):
        vmax, km, c_raw, thermal = p
        return vmax * m / (km + m) + c_raw * np.sqrt(np.maximum(u, 0.0)) + thermal * t
    res = least_squares(lambda p: pred(p) - y, x0=np.array([5.0, 0.8, 1.0, 0.0]), bounds=([0.0, 1e-8, -10.0, -20.0], [20.0, 20.0, 10.0, 20.0]))
    _params = tuple(float(v) for v in res.x)
    return predict

def predict(row):
    vmax, km, c_raw, thermal = _params
    m = float(row['m_2'])
    u = float(row['u_raw'])
    t = float(row['t_proc'])
    return float(vmax * m / (km + m) + c_raw * math.sqrt(max(u, 0.0)) + thermal * t)