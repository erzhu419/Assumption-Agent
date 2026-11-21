import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] for r in rows], dtype=float)
    H = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p, uu, hh):
        af, tf, ass, ts = p
        return uu * (af * (-np.expm1(-hh / tf)) - ass * (-np.expm1(-hh / ts)))
    starts = [[1.0, 1.0, 1.0, 10.0], [2.0, 1.0, 1.0, 20.0], [3.0, 0.5, 1.0, 8.0], [1.0, 2.0, 2.0, 15.0]]
    best = None
    for p0 in starts:
        res = least_squares(lambda p: pred(p, u, H) - y, p0, bounds=([0.0, 1e-4, 0.0, 0.05], [20.0, 20.0, 20.0, 500.0]), max_nfev=30000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    af, tf, ass, ts = p
    h = float(row['H']); u = float(row['u_now'])
    return float(u * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts))))