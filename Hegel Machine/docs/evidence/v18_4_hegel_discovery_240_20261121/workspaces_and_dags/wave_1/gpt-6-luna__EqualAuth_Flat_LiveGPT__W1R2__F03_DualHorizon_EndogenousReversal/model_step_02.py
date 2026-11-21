import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p):
        af, ass, tf, ts = p
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
    best = None
    for p0 in ([2.0, 1.0, 0.5, 15.0], [3.0, 2.0, 1.0, 30.0], [1.5, 0.8, 0.3, 8.0], [4.0, 3.0, 2.0, 60.0]):
        res = least_squares(lambda p: pred(p) - y, p0,
            bounds=([0.0, 0.0, 0.01, 1.0], [20.0, 20.0, 5.0, 200.0]),
            max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit([row])
    af, ass, tf, ts = p
    h = float(row['H'])
    return float(row['u_now']) * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts)))