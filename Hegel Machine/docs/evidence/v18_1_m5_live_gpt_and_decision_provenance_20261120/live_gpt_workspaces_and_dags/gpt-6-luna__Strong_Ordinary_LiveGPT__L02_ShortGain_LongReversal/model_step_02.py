import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    H = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        af, ass, tf, ts = p
        return u * (af * (-np.expm1(-H / tf)) - ass * (-np.expm1(-H / ts)))
    def residual(p):
        return pred(p) - y
    best = None
    for guess in ([1.0, 0.5, 0.3, 8.0], [0.8, 0.5, 1.0, 15.0], [2.0, 1.0, 0.5, 12.0]):
        res = least_squares(residual, guess, bounds=([0, 0, 1e-4, 0.5001], [20, 20, 20, 200]))
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    af, ass, tf, ts = fit(_TRAIN_ROWS)
    H, u = float(row['H']), float(row['u_now'])
    return float(u * (af * (-math.expm1(-H / tf)) - ass * (-math.expm1(-H / ts))))
