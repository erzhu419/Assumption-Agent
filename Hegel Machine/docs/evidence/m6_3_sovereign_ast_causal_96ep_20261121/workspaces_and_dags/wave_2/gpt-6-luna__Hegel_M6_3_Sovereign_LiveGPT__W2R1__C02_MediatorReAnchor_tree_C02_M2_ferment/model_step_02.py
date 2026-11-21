import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_raw']) for r in train_rows], dtype=float)
    s = np.array([float(r['s_surf']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(p):
        vmax, km, c_raw = p
        return vmax * s / (km + s) + c_raw * np.sqrt(np.maximum(u, 0.0))
    res = least_squares(lambda p: pred(p) - y, x0=np.array([5.0, 0.8, 1.0]), bounds=([0.0, 1e-8, -10.0], [20.0, 20.0, 10.0]))
    _params = tuple(float(v) for v in res.x)
    return predict

def predict(row):
    vmax, km, c_raw = _params
    u = float(row['u_raw'])
    s = float(row['s_surf'])
    return float(vmax * s / (km + s) + c_raw * math.sqrt(max(u, 0.0)))