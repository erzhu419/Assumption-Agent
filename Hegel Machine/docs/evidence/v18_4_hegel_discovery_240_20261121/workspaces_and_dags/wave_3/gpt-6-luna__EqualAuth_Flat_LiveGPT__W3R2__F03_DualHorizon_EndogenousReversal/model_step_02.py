import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, uu, hh):
        af, tf, ass, ts = p
        return uu * (af * (-np.expm1(-hh / tf)) - ass * (-np.expm1(-hh / ts)))
    def resid(p):
        return pred(p, u, h) - y
    best = None
    for seed in ([1.5, 0.3, 1.0, 10.0], [1.0, 0.2, 0.8, 20.0], [2.0, 0.5, 1.2, 8.0], [1.0, 1.0, 1.0, 15.0]):
        res = least_squares(resid, seed, bounds=([0, 0.005, 0, 0.6], [10, 5, 10, 200]), max_nfev=30000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2): best = res
    return best.x

def predict(row):
    p = fit(_train_rows)
    af, tf, ass, ts = p
    h, u = float(row['H']), float(row['u_now'])
    return float(u * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts))))
