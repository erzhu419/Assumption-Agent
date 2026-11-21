import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    s = np.array([float(r['s_surf']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_raw']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, km, c_raw = p
        return vmax * s / (km + s) + c_raw * np.sqrt(np.maximum(u, 0.0)) - y
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.2]), bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]))
    return {'vmax': float(result.x[0]), 'k_m': float(result.x[1]), 'c_raw': float(result.x[2])}

def predict(row):
    p = fit_params
    s = float(row['s_surf'])
    u = max(float(row['u_raw']), 0.0)
    return float(p['vmax'] * s / (p['k_m'] + s) + p['c_raw'] * math.sqrt(u))