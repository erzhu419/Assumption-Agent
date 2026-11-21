import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    m2 = np.array([float(r['m_2']) for r in rows])
    root_raw = np.array([math.sqrt(float(r['u_raw'])) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def residuals(p):
        vmax, km, c_raw = p
        return vmax * m2 / (km + m2) + c_raw * root_raw - y

    _params = least_squares(residuals, x0=[8.3, 1.35, 0.32], bounds=([0.0, 1e-9, 0.0], np.inf)).x
    return predict

def predict(row) -> float:
    vmax, km, c_raw = _params
    m2 = float(row['m_2'])
    return float(vmax * m2 / (km + m2) + c_raw * math.sqrt(float(row['u_raw'])))