import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    af, ass, tf, ts, g = p
    h = max(float(row['H']), 0.0)
    u = float(row['u_now'])
    sg = float(row['s_grad'])
    gain = af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts))
    return u * gain - g * sg

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - ys
    starts = [np.array([1.0, 1.0, 0.5, 10.0, 0.0]), np.array([2.0, 2.0, 1.0, 20.0, 0.1])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 0.0, 1e-4, 0.5001, -10.0], [20.0, 20.0, 0.5, 1000.0, 10.0]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_pred(row, p))