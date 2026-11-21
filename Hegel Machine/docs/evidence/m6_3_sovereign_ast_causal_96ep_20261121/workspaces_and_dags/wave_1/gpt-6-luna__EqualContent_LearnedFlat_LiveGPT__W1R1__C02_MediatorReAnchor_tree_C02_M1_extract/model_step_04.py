import math
import numpy as np
from scipy.optimize import least_squares

_state = None

def _features(row):
    return (float(row['s_surf']), math.sqrt(max(float(row['u_raw']), 0.0)), float(row['t_proc']), float(row['m_2']))

def fit(train_rows):
    global _state
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(p, x):
        vmax, km, c_raw, c_t, c_m2 = p
        s, sqrt_u, t, m2 = x
        return vmax * s / (km + s) + c_raw * sqrt_u + c_t * t + c_m2 * m2
    def residual(p):
        return np.array([pred(p, x) for x in X]) - y
    best = None
    for vmax0 in (4.0, 8.0, 12.0):
        for km0 in (0.3, 1.0, 3.0):
            p0 = np.array([vmax0, km0, 1.0, 0.0, 0.0])
            r = least_squares(residual, p0, bounds=([-100, 1e-6, -100, -100, -100], [100, 100, 100, 100, 100]), max_nfev=20000)
            score = float(np.mean(residual(r.x) ** 2))
            if best is None or score < best[0]:
                best = (score, r.x)
    _state = best[1]
    return predict

def predict(row):
    vmax, km, c_raw, c_t, c_m2 = _state
    s, sqrt_u, t, m2 = _features(row)
    return float(vmax * s / (km + s) + c_raw * sqrt_u + c_t * t + c_m2 * m2)