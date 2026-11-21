import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return u, s

def fit(train_rows):
    global _params
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows], dtype=float)
    def pred_vec(p):
        vmax, rho, c = p
        vals = []
        for r in rows:
            u, s = _features(r)
            vals.append(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * np.sqrt(u))
        return np.asarray(vals)
    best = None
    for start in ([6.0, 0.7, 0.0], [8.0, 0.2, 0.5], [4.0, 0.9, -0.5]):
        res = least_squares(lambda p: pred_vec(p) - ys, x0=np.asarray(start, dtype=float), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
        score = float(np.sum(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x)
    _params = best[1]
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, rho, c = _params
    u, s = _features(row)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * np.sqrt(u))