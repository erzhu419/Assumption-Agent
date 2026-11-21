import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    af, ass, tf, ts = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts)))

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def resid(p):
        return np.array([_pred(r, p) for r in train_rows]) - y
    best = None
    for init in ([1.0, 1.0, 1.0, 20.0], [1.0, 1.5, 0.5, 15.0], [2.0, 1.0, 1.0, 10.0], [0.8, 1.2, 0.2, 30.0]):
        result = least_squares(resid, init, bounds=([0.0, 0.0, 1e-4, 1e-4], [20.0, 20.0, 100.0, 200.0]), max_nfev=30000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_pred(row, p))