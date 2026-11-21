import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    m, u, y = np.asarray([(r['m_2'], r['u_raw'], r['y']) for r in train_rows], dtype=float).T
    raw = np.sqrt(u)

    def residual(p):
        vmax, km, c_raw = p
        return vmax * m / (km + m) + c_raw * raw - y

    _params = least_squares(residual, [7.0, 1.0, 0.3], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row):
    vmax, km, c_raw = _params
    m = float(row['m_2'])
    return float(vmax * m / (km + m) + c_raw * math.sqrt(row['u_raw']))
