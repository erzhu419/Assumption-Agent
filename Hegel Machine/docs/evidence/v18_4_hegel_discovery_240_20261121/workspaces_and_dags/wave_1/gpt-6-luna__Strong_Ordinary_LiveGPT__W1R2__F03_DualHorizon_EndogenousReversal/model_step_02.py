import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    def pred(p, r):
        af, ass, tf, ts = p
        h = float(r['H'])
        u = float(r['u_now'])
        return u * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts)))
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in rows])
    best = None
    for seed in ([1.0, 1.0, 1.0, 10.0], [2.0, 2.0, 0.5, 20.0], [1.0, 1.5, 2.0, 15.0], [3.0, 3.0, 0.8, 30.0]):
        sol = least_squares(residual, np.array(seed, dtype=float), bounds=([0, 0, 1e-5, 1e-5], [100, 100, 1000, 1000]), max_nfev=30000)
        if best is None or np.sum(sol.fun**2) < np.sum(best.fun**2):
            best = sol
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    af, ass, tf, ts = p
    h = float(row['H'])
    u = float(row['u_now'])
    return float(u * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts))))
