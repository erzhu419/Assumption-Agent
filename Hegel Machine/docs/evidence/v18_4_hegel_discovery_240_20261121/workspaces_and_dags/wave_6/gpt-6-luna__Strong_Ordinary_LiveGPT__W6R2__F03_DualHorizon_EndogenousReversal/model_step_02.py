import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        af, ass, tf, ts = p
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
    best = None
    for start in ([1.0, 1.0, .3, 15.0], [1.2, .8, .5, 25.0], [2.0, 1.0, 1.0, 12.0], [.8, .7, .2, 40.0]):
        res = least_squares(lambda p: pred(p) - y, start, bounds=([0, 0, 1e-4, 1e-3], [20, 20, 100, 500]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    af, ass, tf, ts = fit(_TRAIN_ROWS)
    h = float(row['H']); u = float(row['u_now'])
    return u * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts)))