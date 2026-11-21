import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    h = np.array([float(r['H']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(p):
        af, ass, tf, ts = p
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
    def residual(p):
        return pred(p) - y
    starts = [[1.5, 1.0, 0.5, 12.0], [2.0, 1.5, 1.0, 20.0], [1.0, 0.8, 0.2, 8.0]]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0, 0, 1e-5, 0.500001], [100, 100, 0.499999, 1e4]))
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    af, ass, tf, ts = fit(train_rows)
    h = float(row['H'])
    u = float(row['u_now'])
    return float(u * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts))))