import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    H = np.array([float(r['H']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    z = y / np.maximum(np.abs(u), 1e-8)
    def pred(p):
        af, ass, tf, ts = p[0], p[1], math.exp(p[2]), math.exp(p[3])
        return af * (1.0 - np.exp(-H / tf)) - ass * (1.0 - np.exp(-H / ts))
    def resid(p):
        return pred(p) - z
    starts = [
        [2.0, 1.0, math.log(0.5), math.log(10.0)],
        [3.0, 1.5, math.log(1.0), math.log(20.0)],
        [1.5, 0.8, math.log(0.3), math.log(5.0)],
        [4.0, 2.0, math.log(2.0), math.log(30.0)],
    ]
    best = None
    for p0 in starts:
        res = least_squares(resid, p0, max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x.tolist()

def predict(row):
    af, ass, log_tf, log_ts = fit_params
    tf, ts = math.exp(log_tf), math.exp(log_ts)
    H = float(row['H'])
    return float(row['u_now']) * (af * (1.0 - math.exp(-H / tf)) - ass * (1.0 - math.exp(-H / ts)))