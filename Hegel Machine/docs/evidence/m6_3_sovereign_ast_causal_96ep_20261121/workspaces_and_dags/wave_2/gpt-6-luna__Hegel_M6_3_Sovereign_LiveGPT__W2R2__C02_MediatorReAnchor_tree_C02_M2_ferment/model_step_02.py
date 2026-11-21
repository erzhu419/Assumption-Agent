import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    s = np.array([float(r['s_surf']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_raw']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, km, c = p
        return vmax * s / (km + s) + c * np.sqrt(np.maximum(u, 0.0)) - y
    result = least_squares(residual, x0=np.array([7.0, 1.0, 1.0]), bounds=([0.0, 1e-8, -20.0], [100.0, 100.0, 20.0]))
    _params = result.x
    return predict

def predict(row):
    vmax, km, c = _params
    s = float(row['s_surf'])
    u = max(float(row['u_raw']), 0.0)
    return float(vmax * s / (km + s) + c * math.sqrt(u))