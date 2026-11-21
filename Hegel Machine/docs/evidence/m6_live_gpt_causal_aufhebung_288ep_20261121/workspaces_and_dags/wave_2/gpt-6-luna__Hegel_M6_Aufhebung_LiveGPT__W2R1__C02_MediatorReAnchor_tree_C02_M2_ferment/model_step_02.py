import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    m = np.asarray([float(r['m_2']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_raw']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)

    def residual(p):
        vmax, km, c_raw = p
        return vmax * m / (km + m) + c_raw * np.sqrt(np.maximum(u, 0.0)) - y

    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.5]),
                           bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, km, c_raw = _params
    m2 = float(row['m_2'])
    u = max(float(row['u_raw']), 0.0)
    return float(vmax * m2 / (km + m2) + c_raw * math.sqrt(u))