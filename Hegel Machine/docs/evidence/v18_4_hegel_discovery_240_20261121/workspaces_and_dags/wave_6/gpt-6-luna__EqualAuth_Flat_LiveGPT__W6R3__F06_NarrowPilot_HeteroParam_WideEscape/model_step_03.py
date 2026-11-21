import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(row, p):
    vmax, km, c_loc, k_sink = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * u / (km + u) + c_loc * math.sqrt(u) - k_sink * s * u

def fit(train_rows):
    global _PARAMS
    if not train_rows:
        raise ValueError('train_rows must not be empty')
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([3.0, 0.8, 0.2, 0.5]), bounds=([0.0, 1e-8, -5.0, -5.0], [20.0, 20.0, 5.0, 5.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(row, _PARAMS))