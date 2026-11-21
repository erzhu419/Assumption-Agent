import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    m = np.asarray([float(r['m_2']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, km = p
        return vmax * m / (km + m) - y
    result = least_squares(residual, x0=[8.0, 1.0], bounds=([0.0, 1e-8], [100.0, 100.0]))
    _params = result.x
    return predict

def predict(row):
    vmax, km = _params
    m = float(row['m_2'])
    return float(vmax * m / (km + m))