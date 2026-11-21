import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    g = np.array([r['s_grad'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        af, ass, tf, ts, cg = p
        return u * (af * (1.0 - np.exp(-h / tf)) - ass * (1.0 - np.exp(-h / ts))) + cg * g
    best = None
    for p0 in ([1.0, 1.0, 1.0, 20.0, 0.0], [2.0, 2.0, 0.5, 30.0, 0.3], [1.0, 2.0, 2.0, 15.0, -0.3], [3.0, 3.0, 0.2, 40.0, 0.0]):
        res = least_squares(lambda p: pred(p) - y, p0, bounds=([-20.0, -20.0, 0.02, 0.1, -20.0], [20.0, 20.0, 50.0, 200.0, 20.0]), max_nfev=30000)
        loss = float(np.sum(res.fun ** 2))
        if best is None or loss < best[0]:
            best = (loss, res.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise ValueError('fit(train_rows) must be called before predict(row)')
    af, ass, tf, ts, cg = _PARAMS
    h = float(row['H'])
    return float(float(row['u_now']) * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts))) + cg * float(row['s_grad']))