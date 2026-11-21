import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    H = np.array([r['H'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p):
        af, ass, tf, ts = p
        return u * (af * (1.0 - np.exp(-H / tf)) - ass * (1.0 - np.exp(-H / ts)))
    starts = [[1.0, 1.0, 1.0, 10.0], [2.0, 2.0, 0.5, 20.0], [1.0, 2.0, 2.0, 30.0]]
    fits = [least_squares(lambda p: pred(p) - y, x0=s, bounds=([0, 0, 1e-4, 1e-4], [20, 20, 1e3, 1e3]), max_nfev=20000) for s in starts]
    best = min(fits, key=lambda q: float(np.sum(q.fun ** 2)))
    return best.x

def predict(row):
    af, ass, tf, ts = fit(train_rows)
    h = float(row['H'])
    return float(row['u_now']) * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts)))