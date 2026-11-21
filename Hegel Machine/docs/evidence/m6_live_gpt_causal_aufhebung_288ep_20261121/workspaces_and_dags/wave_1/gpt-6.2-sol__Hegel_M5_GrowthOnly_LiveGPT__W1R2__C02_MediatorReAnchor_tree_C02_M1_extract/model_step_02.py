import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    m = np.array([float(r['m_2']) for r in rows])
    root_u = np.sqrt(np.array([float(r['u_raw']) for r in rows]))
    y = np.array([float(r['y']) for r in rows])

    def residual(p):
        vmax, km, c_raw = p
        return vmax * m / (km + m) + c_raw * root_u - y

    _params = tuple(least_squares(residual, [8.0, 1.0, 0.3], bounds=([0.0, 1e-9, 0.0], np.inf)).x)
    return _params

def predict(row) -> float:
    vmax, km, c_raw = _params
    m = float(row['m_2'])
    return float(vmax * m / (km + m) + c_raw * math.sqrt(float(row['u_raw'])))
