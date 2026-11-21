import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    m = np.array([float(r['m_2']) for r in rows])
    raw = np.array([math.sqrt(float(r['u_raw'])) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def residual(p):
        vmax, k, c_raw = p
        return vmax * m / (k + m) + c_raw * raw - y

    _params = least_squares(residual, (7.4, 1.3, 0.28), bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row):
    vmax, k, c_raw = _params
    m = float(row['m_2'])
    return float(vmax * m / (k + m) + c_raw * math.sqrt(float(row['u_raw'])))
