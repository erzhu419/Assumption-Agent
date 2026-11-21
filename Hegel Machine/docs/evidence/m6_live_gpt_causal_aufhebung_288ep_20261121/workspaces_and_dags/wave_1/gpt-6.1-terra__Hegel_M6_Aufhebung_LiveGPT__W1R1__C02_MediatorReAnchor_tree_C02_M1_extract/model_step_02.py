import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    m = np.array([float(r['m_2']) for r in train_rows])
    u = np.array([float(r['u_raw']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        vmax, km, c = p
        return vmax * m / (km + m) + c * np.sqrt(u) - y
    result = least_squares(residual, [6.0, 1.0, 0.5], bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]))
    _params = result.x

def predict(row):
    vmax, km, c = _params
    m = float(row['m_2'])
    return float(vmax * m / (km + m) + c * math.sqrt(float(row['u_raw'])))
