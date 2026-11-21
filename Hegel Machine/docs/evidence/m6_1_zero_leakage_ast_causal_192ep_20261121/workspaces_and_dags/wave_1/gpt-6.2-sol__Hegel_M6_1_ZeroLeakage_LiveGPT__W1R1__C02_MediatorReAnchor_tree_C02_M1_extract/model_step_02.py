import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['m_2'], math.sqrt(float(r['u_raw'])), r['y']) for r in train_rows], dtype=float)
    m, root_u, y = data.T

    def residual(p):
        vmax, km, c_raw = p
        return vmax * m / (km + m) + c_raw * root_u - y

    _params = least_squares(residual, [8.0, 1.0, 0.3], bounds=([0.0, 1e-9, 0.0], np.inf)).x
    return _params

def predict(row) -> float:
    vmax, km, c_raw = _params
    m = float(row['m_2'])
    return float(vmax * m / (km + m) + c_raw * math.sqrt(float(row['u_raw'])))