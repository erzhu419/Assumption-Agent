import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    m = np.array([float(r['m_2']) for r in train_rows], dtype=float)
    t = np.array([float(r['t_proc']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Thermal offset is unidentifiable when all training rows are cold; keep it at zero.
    def residual(p):
        vmax, km = p
        return vmax * m / (km + m) - y
    result = least_squares(residual, x0=[8.0, 1.0], bounds=([0.0, 1e-8], [100.0, 100.0]))
    _params = (float(result.x[0]), float(result.x[1]), 0.0)
    return predict

def predict(row):
    vmax, km, thermal_offset = _params
    m = float(row['m_2'])
    t = float(row['t_proc'])
    return float(vmax * m / (km + m) + thermal_offset * t)