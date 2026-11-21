import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([(r['m_2'], math.sqrt(r['u_raw']), r['y']) for r in train_rows], dtype=float)
    m, root_u, y = data.T
    def residual(p):
        vmax, km, c_raw = p
        return vmax * m / (km + m) + c_raw * root_u - y
    _params = least_squares(residual, (7.0, 1.0, 0.3), bounds=([0.0, 1e-8, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row):
    vmax, km, c_raw = _params
    m = row['m_2']
    return float(vmax * m / (km + m) + c_raw * math.sqrt(row['u_raw']))
