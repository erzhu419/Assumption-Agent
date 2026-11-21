import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row, km):
    s = float(row['s_surf'])
    u = float(row['u_raw'])
    t = float(row['t_proc'])
    return np.array([s / (km + s), math.sqrt(max(u, 0.0)), t * float(row['m_1']), t * float(row['m_2']), t * float(row['m_3'])], dtype=float)

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        km = math.exp(p[0])
        X = np.vstack([_features(r, km) for r in train_rows])
        return X @ p[1:] - y
    best = None
    for km0 in (0.15, 0.5, 1.5, 4.0):
        p0 = np.array([math.log(km0), 5.0, 0.5, 0.0, 0.0, 0.0])
        result = least_squares(residual, p0, max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _PARAMS = best.x.copy()
    return predict

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    km = math.exp(float(_PARAMS[0]))
    return float(_features(row, km) @ _PARAMS[1:])