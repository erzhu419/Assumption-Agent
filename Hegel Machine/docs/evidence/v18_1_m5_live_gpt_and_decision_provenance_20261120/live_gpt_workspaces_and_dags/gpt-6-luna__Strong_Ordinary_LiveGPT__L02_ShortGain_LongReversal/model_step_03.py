import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        af, ass, tf, ts = p
        return u * (af * (1.0 - np.exp(-h / tf)) - ass * (1.0 - np.exp(-h / ts)))
    best = None
    for start in ([1.0, 1.0, 0.5, 10.0], [2.0, 1.0, 1.0, 20.0], [1.0, 0.5, 0.2, 5.0], [3.0, 2.0, 2.0, 30.0]):
        res = least_squares(lambda p: pred(p) - y, x0=np.array(start), bounds=([0.0, 0.0, 1e-4, 1e-3], [20.0, 20.0, 10.0, 200.0]), max_nfev=30000)
        score = float(np.sum(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x)
    _PARAMS = np.asarray(best[1], dtype=float)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    af, ass, tf, ts = _PARAMS
    h = float(row['H'])
    u = float(row['u_now'])
    return float(u * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts))))