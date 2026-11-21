import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _pred(row, p):
    af, tf, ass, ts = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def residual(p):
        return np.asarray([_pred(r, p) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=[1.0, 0.3, 1.0, 15.0], bounds=([0.0, 1e-5, 0.0, 1e-4], [10.0, 10.0, 10.0, 1e3]), max_nfev=50000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_pred(row, _PARAMS))