import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_KM = None

def _features(row, km):
    u = max(float(row['u_raw']), 0.0)
    t = float(row['t_proc'])
    s = float(row['s_surf'])
    m1 = float(row['m_1'])
    m2 = float(row['m_2'])
    m3 = float(row['m_3'])
    # Cold-regime surface saturation and raw-input response, with
    # processing-gated mediator contributions and a quadratic-time term.
    return np.array([
        s / (km + s), math.sqrt(u),
        t*m1, t*m2, t*m3,
        t*t, t*t*m1, t*t*m2, t*t*m3
    ], dtype=float)

def fit(train_rows):
    global _PARAMS, _KM
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    best = None
    for km0 in (0.15, 0.5, 1.5, 4.0):
        def residual(p):
            km = math.exp(float(p[0]))
            X = np.vstack([_features(r, km) for r in train_rows])
            return X @ p[1:] - y
        p0 = np.array([math.log(km0), 5.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = least_squares(residual, p0, max_nfev=20000)
        score = float(np.sum(result.fun**2))
        if best is None or score < best[0]:
            best = (score, result.x.copy())
    _KM = math.exp(float(best[1][0]))
    _PARAMS = best[1][1:].copy()
    return predict

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_features(row, _KM) @ _PARAMS)