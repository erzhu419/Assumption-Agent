import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    m = np.array([r['m_2'] for r in train_rows], dtype=float)
    raw = np.sqrt(np.array([r['u_raw'] for r in train_rows], dtype=float))
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        vmax, km, c_raw = p
        return vmax * m / (km + m) + c_raw * raw - y

    _params = least_squares(residual, [7.4, 1.325, 0.283], bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    vmax, km, c_raw = _params
    m = row['m_2']
    return float(vmax * m / (km + m) + c_raw * math.sqrt(row['u_raw']))
