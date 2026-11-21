import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        af, a_s, tf, ts = p
        return u * (af * (-np.expm1(-H / tf)) - a_s * (-np.expm1(-H / ts)))
    def residual(p):
        return pred(p) - y
    best = None
    for start in ([1.0, 0.5, 1.0, 15.0], [1.5, 1.0, 0.5, 20.0], [2.0, 1.0, 2.0, 30.0], [1.0, 0.5, 0.2, 10.0]):
        res = least_squares(residual, start, bounds=([0, 0, 1e-5, 1e-4], [20, 20, 100, 500]), max_nfev=30000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(train_rows)
    af, a_s, tf, ts = p
    H = float(row['H'])
    u = float(row['u_now'])
    return float(u * (af * (-math.expm1(-H / tf)) - a_s * (-math.expm1(-H / ts))))