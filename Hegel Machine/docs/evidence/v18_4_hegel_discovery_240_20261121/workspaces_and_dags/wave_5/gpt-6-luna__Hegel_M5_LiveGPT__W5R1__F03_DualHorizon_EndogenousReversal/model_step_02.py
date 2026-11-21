import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows])
    H = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def predict_params(p):
        af, ass = p[0], p[1]
        tf = math.exp(p[2])
        ts = tf + math.exp(p[3])
        return u * (af * (1.0 - np.exp(-H / tf)) - ass * (1.0 - np.exp(-H / ts)))
    def residual(p):
        return predict_params(p) - y
    starts = [np.array([1.5, 1.5, math.log(0.5), math.log(10.0)]), np.array([2.0, 1.0, math.log(1.0), math.log(15.0)]), np.array([1.0, 2.0, math.log(0.2), math.log(20.0)])]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, max_nfev=20000)
        score = float(np.sum(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x)
    p = best[1]
    return {'amp_fast': float(p[0]), 'amp_slow': float(p[1]), 'tau_fast': float(math.exp(p[2])), 'tau_slow': float(math.exp(p[2]) + math.exp(p[3]))}

def predict(row):
    p = fit_globals
    tf, ts = p['tau_fast'], p['tau_slow']
    return float(row['u_now'] * (p['amp_fast'] * (1.0 - math.exp(-float(row['H']) / tf)) - p['amp_slow'] * (1.0 - math.exp(-float(row['H']) / ts))))