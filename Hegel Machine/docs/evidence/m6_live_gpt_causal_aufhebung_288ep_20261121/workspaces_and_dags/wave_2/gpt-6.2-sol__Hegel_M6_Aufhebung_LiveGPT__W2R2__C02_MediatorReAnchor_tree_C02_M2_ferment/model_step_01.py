import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    m = np.array([r['m_2'] for r in rows], dtype=float)
    raw = np.array([math.sqrt(r['u_raw']) for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        vmax, km, c_raw = p
        return vmax * m / (km + m) + c_raw * raw - y

    _params = least_squares(residual, [7.537, 1.421, 0.355], bounds=([0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row):
    vmax, km, c_raw = _params
    m = row['m_2']
    return float(vmax * m / (km + m) + c_raw * math.sqrt(row['u_raw']))
