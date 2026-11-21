import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    H = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred_log(p):
        af, tf, ass, ts = np.exp(p)
        return u * (af * (-np.expm1(-H / tf)) - ass * (-np.expm1(-H / ts)))
    starts = [np.log([1.5, 0.2, 1.0, 10.0]), np.log([1.0, 0.5, 1.0, 20.0]), np.log([2.0, 0.1, 1.0, 5.0])]
    best = None
    for p0 in starts:
        res = least_squares(lambda p: pred_log(p) - y, p0, max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    af, tf, ass, ts = np.exp(best.x)
    return {'amp_fast': float(af), 'tau_fast': float(tf), 'amp_slow': float(ass), 'tau_slow': float(ts)}

def predict(row):
    af = row['amp_fast']
    tf = row['tau_fast']
    ass = row['amp_slow']
    ts = row['tau_slow']
    u = float(row['u_now'])
    H = float(row['H'])
    return float(u * (af * (-math.expm1(-H / tf)) - ass * (-math.expm1(-H / ts))))