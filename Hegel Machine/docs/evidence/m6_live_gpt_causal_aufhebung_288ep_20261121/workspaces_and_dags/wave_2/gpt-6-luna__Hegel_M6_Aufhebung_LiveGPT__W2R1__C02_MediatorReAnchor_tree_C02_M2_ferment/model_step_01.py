import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    m2 = np.array([float(r['m_2']) for r in train_rows])
    u = np.array([float(r['u_raw']) for r in train_rows])
    def residual(p):
        vmax, km, c_raw = p
        return vmax * m2 / (km + m2) + c_raw * np.sqrt(u) - y
    result = least_squares(residual, x0=np.array([7.0, 1.0, 0.5]), bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]))
    return tuple(float(v) for v in result.x)

def predict(row):
    vmax, km, c_raw = fit_rows_params
    m2 = float(row['m_2'])
    u = float(row['u_raw'])
    return float(vmax * m2 / (km + m2) + c_raw * math.sqrt(u))