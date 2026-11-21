import math
import numpy as np
from scipy.optimize import least_squares

fit_params = None

def fit(train_rows):
    global fit_params
    rows = train_rows
    H = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        af, ass, log_tf, log_ts = p
        tf, ts = math.exp(log_tf), math.exp(log_ts)
        pred = u * (af * (1.0 - np.exp(-H / tf)) - ass * (1.0 - np.exp(-H / ts)))
        return pred - y
    starts = [[2.0, 1.0, math.log(0.5), math.log(8.0)], [1.0, 2.0, math.log(0.3), math.log(10.0)], [3.0, 2.0, math.log(1.0), math.log(20.0)]]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.0, 0.0, math.log(0.01), math.log(0.05)], [20.0, 20.0, math.log(10.0), math.log(200.0)]), max_nfev=20000)
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    fit_params = best.x
    return fit_params

def predict(row):
    if fit_params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    af, ass, log_tf, log_ts = fit_params
    H = float(row['H'])
    tf, ts = math.exp(log_tf), math.exp(log_ts)
    return float(row['u_now']) * (af * (1.0 - math.exp(-H / tf)) - ass * (1.0 - math.exp(-H / ts)))