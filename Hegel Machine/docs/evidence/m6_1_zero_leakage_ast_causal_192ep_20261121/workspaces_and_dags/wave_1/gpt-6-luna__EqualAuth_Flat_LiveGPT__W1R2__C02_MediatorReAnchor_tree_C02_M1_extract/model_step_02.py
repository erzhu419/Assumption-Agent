import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(float(row.get('u_raw', 0.0)), 0.0)
    s = max(float(row.get('s_surf', 0.0)), 0.0)
    t = max(float(row.get('t_proc', 0.0)), 0.0)
    m1 = float(row.get('m_1', 0.0))
    return u, s, t, m1

def _predict_params(p, row):
    vmax, km, c_raw, a_t, b_tm = p
    u, s, t, m1 = _features(row)
    sat = vmax * s / (km + s + 1e-12)
    return sat + c_raw * math.sqrt(u) + t * (a_t + b_tm * m1)

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([_features(r) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.array([_predict_params(p, r) for r in rows]) - y
    # Positive saturation parameters; thermal response is unconstrained.
    best = None
    for start in ([8.0, 1.0, 1.0, -1.0, -0.2], [10.0, 2.0, 0.5, 0.0, -0.5]):
        res = least_squares(residual, start, bounds=([0.0, 1e-6, -20.0, -20.0, -20.0], [30.0, 30.0, 20.0, 20.0, 20.0]), max_nfev=10000)
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_predict_params(p, row))