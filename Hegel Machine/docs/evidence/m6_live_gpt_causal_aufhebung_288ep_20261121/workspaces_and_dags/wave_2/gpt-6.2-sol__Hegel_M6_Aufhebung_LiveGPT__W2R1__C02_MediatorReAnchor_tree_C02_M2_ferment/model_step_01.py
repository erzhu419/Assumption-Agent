import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    m = np.array([r['m_2'] for r in rows], dtype=float)
    root_u = np.array([math.sqrt(r['u_raw']) for r in rows])
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        vmax, km, c_raw = p
        return vmax * m / (km + m) + c_raw * root_u - y

    _params = least_squares(residual, [max(1.0, float(y.max())), 1.0, 0.5], bounds=([0.0, 1e-9, 0.0], np.inf)).x

def predict(row):
    vmax, km, c_raw = _params
    m = row['m_2']
    return float(vmax * m / (km + m) + c_raw * math.sqrt(row['u_raw']))
