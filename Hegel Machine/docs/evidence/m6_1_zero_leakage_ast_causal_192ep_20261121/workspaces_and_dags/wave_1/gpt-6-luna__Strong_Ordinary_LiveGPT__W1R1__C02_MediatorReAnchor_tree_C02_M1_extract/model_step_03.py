import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(p, row):
    vmax, k, c_raw, c_t = p
    m = max(0.0, float(row.get('m_2', 0.0)))
    u = max(0.0, float(row.get('u_raw', 0.0)))
    t = float(row.get('t_proc', 0.0))
    return vmax * m / (k + m + 1e-12) + c_raw * math.sqrt(u) - c_t * t

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(p, r) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([7.0, 1.0, 0.5, 0.5]), bounds=([0.0, 1e-6, -10.0, -10.0], [100.0, 100.0, 10.0, 10.0]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(_PARAMS, row))