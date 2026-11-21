import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_raw']) for r in train_rows], dtype=float)
    m = np.array([float(r['m_2']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, km, c = p
        return vmax * m / (km + m) + c * np.sqrt(np.maximum(u, 0.0)) - y
    result = least_squares(residual, x0=np.array([6.0, 1.0, 0.5]), bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, km, c = _params
    m = float(row['m_2'])
    u = max(float(row['u_raw']), 0.0)
    return float(vmax * m / (km + m) + c * math.sqrt(u))