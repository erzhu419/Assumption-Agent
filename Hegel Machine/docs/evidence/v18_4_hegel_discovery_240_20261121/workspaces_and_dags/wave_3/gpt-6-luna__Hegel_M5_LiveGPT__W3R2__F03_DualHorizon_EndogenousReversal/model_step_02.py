import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    U = np.array([float(r['u_now']) for r in train_rows])
    H = np.array([float(r['H']) for r in train_rows])
    Y = np.array([float(r['y']) for r in train_rows])
    def pred(p, u, h):
        af, ass, tf, ts = p
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
    def residual(p):
        return pred(p, U, H) - Y
    starts = [np.array([1.5, 1.0, 0.3, 10.0]), np.array([2.0, 1.0, 0.2, 20.0]), np.array([1.0, 2.0, 0.5, 15.0])]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0,0,1e-5,0.50001],[20,20,5,200]), max_nfev=30000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(train_rows)
    u = float(row['u_now']); h = float(row['H'])
    af, ass, tf, ts = p
    return float(u * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts))))