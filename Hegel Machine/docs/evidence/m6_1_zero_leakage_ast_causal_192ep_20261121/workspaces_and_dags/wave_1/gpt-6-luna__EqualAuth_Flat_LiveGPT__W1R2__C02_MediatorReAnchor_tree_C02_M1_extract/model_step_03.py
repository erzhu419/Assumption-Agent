import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_raw'])
    s = float(row['s_surf'])
    t = float(row['t_proc'])
    m1 = float(row['m_1'])
    return u, s, t, m1

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    u = np.array([_features(r)[0] for r in rows], dtype=float)
    s = np.array([_features(r)[1] for r in rows], dtype=float)
    t = np.array([_features(r)[2] for r in rows], dtype=float)
    m1 = np.array([_features(r)[3] for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)

    def residual(p):
        vmax, km, craw, beta = p
        pred = vmax * s / (km + s) + craw * np.sqrt(np.maximum(u, 0.0)) + beta * t * m1
        return pred - y

    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.5, -0.8]), bounds=([0.0, 1e-8, -10.0, -10.0], [30.0, 30.0, 10.0, 10.0]))
    _PARAMS = tuple(float(x) for x in result.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    vmax, km, craw, beta = _PARAMS
    u, s, t, m1 = _features(row)
    return float(vmax * s / (km + s) + craw * math.sqrt(max(u, 0.0)) + beta * t * m1)