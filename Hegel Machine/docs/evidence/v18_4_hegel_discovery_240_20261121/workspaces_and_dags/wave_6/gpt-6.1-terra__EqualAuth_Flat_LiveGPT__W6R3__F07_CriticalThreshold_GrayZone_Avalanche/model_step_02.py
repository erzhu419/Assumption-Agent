import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(rows, p):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    vmax, vrho, c, k, critical = p
    return (vmax - vrho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(u) - k * np.maximum(0.0, s - critical) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [r['y'] for r in rows]
    best = None
    for critical in (2.5, 2.58, 2.66):
        result = least_squares(lambda p: _values(rows, p) - y, [5.0, 1.0, 1.0, 2.0, critical], bounds=([0.0, 0.0, -np.inf, 0.0, 2.44], [np.inf, np.inf, np.inf, np.inf, 2.78]), max_nfev=10000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    return float(_values([row], _params)[0])