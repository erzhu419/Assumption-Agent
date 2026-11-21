import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([float(r['u_now']) for r in train_rows])
    h = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        af, ass, tf, ts = np.exp(p)
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
    best = None
    for start in ([math.log(1.0), math.log(0.3), math.log(1.0), math.log(10.0)],
                  [math.log(2.0), math.log(0.2), math.log(0.5), math.log(20.0)],
                  [math.log(0.5), math.log(0.1), math.log(2.0), math.log(30.0)]):
        res = least_squares(lambda p: pred(p) - y, np.array(start), max_nfev=30000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _PARAMS = np.exp(best.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    af, ass, tf, ts = _PARAMS
    h = float(row['H'])
    u = float(row['u_now'])
    return float(u * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts))))