import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row.get('u_raw', 0.0))
    s = float(row.get('s_surf', 0.0))
    t = float(row.get('t_proc', 0.0))
    m1 = float(row.get('m_1', 0.0))
    m2 = float(row.get('m_2', 0.0))
    m3 = float(row.get('m_3', 0.0))
    return u, s, t, m1, m2, m3

def _model(row, p):
    u, s, t, m1, m2, m3 = _features(row)
    vmax, km, c_raw, b_t, b1, b2, b3 = p
    return vmax * s / (km + max(s, 1e-12)) + c_raw * math.sqrt(max(u, 0.0)) + b_t * t + b1 * m1 + b2 * m2 + b3 * m3

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]), bounds=([-100.0, 1e-6, -100.0, -100.0, -100.0, -100.0, -100.0], [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]))
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_model(row, _PARAMS))