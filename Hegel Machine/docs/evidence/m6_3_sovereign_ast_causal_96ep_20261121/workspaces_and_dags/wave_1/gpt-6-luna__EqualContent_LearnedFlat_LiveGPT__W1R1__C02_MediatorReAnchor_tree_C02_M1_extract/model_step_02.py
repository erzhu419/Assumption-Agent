import math
import numpy as np
from scipy.optimize import least_squares

_state = None

def _features(row):
    s = float(row['s_surf'])
    u = float(row['u_raw'])
    t = float(row['t_proc'])
    m1 = float(row['m_1'])
    return s, math.sqrt(max(u, 0.0)), t, m1

def fit(train_rows):
    global _state
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Saturating surface-acidity response plus the raw-input contribution,
    # with a processing term and a separately identified m1 mediator effect.
    def pred(p, x):
        vmax, km, c_raw, c_t, c_m1 = p
        s, sqrt_u, t, m1 = x
        return vmax * s / (km + s) + c_raw * sqrt_u + c_t * t + c_m1 * m1
    def residual(p):
        return np.array([pred(p, x) for x in X]) - y
    p0 = np.array([5.0, 1.0, 0.5, -0.5, 0.5])
    res = least_squares(residual, p0, bounds=([-100, 1e-6, -100, -100, -100], [100, 100, 100, 100, 100]), max_nfev=20000)
    _state = res.x
    return lambda row: predict(row)

def predict(row):
    vmax, km, c_raw, c_t, c_m1 = _state
    s, sqrt_u, t, m1 = _features(row)
    return float(vmax * s / (km + s) + c_raw * sqrt_u + c_t * t + c_m1 * m1)
