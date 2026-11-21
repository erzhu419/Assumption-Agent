import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(float(row.get('u_raw', 0.0)), 0.0)
    t = float(row.get('t_proc', 0.0))
    m2 = max(float(row.get('m_2', 0.0)), 0.0)
    return u, t, m2

def fit(train_rows):
    rows = list(train_rows)
    if not rows:
        return {'p': np.array([6.0, 0.7, 0.5, 0.0])}
    y = np.array([float(r['y']) for r in rows])
    def pred(p, r):
        u, t, m2 = _features(r)
        vmax, km, c_raw, c_proc = p
        return vmax*m2/(km+m2+1e-12) + c_raw*math.sqrt(u) + c_proc*t
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 0.7, 0.5, -1.0]), bounds=([-30.0, 1e-6, -10.0, -10.0], [30.0, 30.0, 10.0, 10.0]))
    return {'p': result.x}

def predict(row):
    p = fit([row])['p']
    u, t, m2 = _features(row)
    vmax, km, c_raw, c_proc = p
    return float(vmax*m2/(km+m2+1e-12) + c_raw*math.sqrt(u) + c_proc*t)
