import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    af, ass, tf, ts = p
    h = float(row['H'])
    return float(row['u_now']) * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts)))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    lo = np.array([0.0, 0.0, 1e-3, 1e-3])
    hi = np.array([20.0, 20.0, 100.0, 100.0])
    best = None
    for af0, ass0, tf0, ts0 in [(2, 2, .2, 2), (3, 3, .5, 5), (5, 5, 1, 10), (2, 1, .1, 20)]:
        p0 = np.array([af0, ass0, tf0, ts0], dtype=float)
        res = least_squares(lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
                            p0, bounds=(lo, hi), max_nfev=20000)
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return _model(row, _PARAMS)
