import math
import numpy as np
from scipy.optimize import least_squares

_MODEL = None

def _features(r):
    u = max(float(r.get('u_raw', 0.0)), 0.0)
    t = max(float(r.get('t_proc', 0.0)), 0.0)
    s = max(float(r.get('s_surf', 0.0)), 0.0)
    m1 = float(r.get('m_1', 0.0))
    m2 = float(r.get('m_2', 0.0))
    m3 = float(r.get('m_3', 0.0))
    return u, t, s, m1, m2, m3

def fit(train_rows):
    global _MODEL
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    x = [_features(r) for r in rows]
    # Parameters: cold saturating surface amplitude/half-point, cold sqrt(raw)
    # term; plus a thermal intercept and independent mediator effects.
    def pred(p, z):
        u, t, s, m1, m2, m3 = z
        vmax, km, craw, bt, b1, b2, b3 = p
        cold = vmax * s / (km + s) + craw * math.sqrt(u)
        return cold + t * (bt + b1*m1 + b2*m2 + b3*m3)
    def residual(p):
        return np.asarray([pred(p, z) for z in x]) - y
    init = np.array([8.0, 0.7, 0.5, 0.0, 0.0, 0.0, 0.0])
    result = least_squares(residual, init, max_nfev=10000)
    _MODEL = result.x
    return _MODEL

def predict(row):
    if _MODEL is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    u, t, s, m1, m2, m3 = _features(row)
    vmax, km, craw, bt, b1, b2, b3 = _MODEL
    cold = vmax * s / (km + s) + craw * math.sqrt(u)
    return float(cold + t * (bt + b1*m1 + b2*m2 + b3*m3))