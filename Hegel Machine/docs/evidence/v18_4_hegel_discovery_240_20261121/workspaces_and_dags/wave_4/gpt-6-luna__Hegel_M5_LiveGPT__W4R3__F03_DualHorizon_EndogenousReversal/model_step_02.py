import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def predict_params(p):
        amp_fast, amp_slow, tau_fast, tau_slow = p
        return u * (amp_fast * (1.0 - np.exp(-h / tau_fast)) - amp_slow * (1.0 - np.exp(-h / tau_slow)))

    best = None
    for start in ([1.0, 1.0, 0.5, 10.0], [2.0, 1.0, 1.0, 20.0], [1.0, 0.5, 0.2, 5.0], [2.0, 1.5, 2.0, 30.0]):
        result = least_squares(lambda p: predict_params(p) - y, x0=np.array(start), bounds=([0.0, 0.0, 1e-4, 1e-3], [100.0, 100.0, 100.0, 1e4]), max_nfev=30000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    p = best.x

    def predict(row):
        amp_fast, amp_slow, tau_fast, tau_slow = p
        H = float(row['H'])
        return float(row['u_now']) * (amp_fast * (1.0 - math.exp(-H / tau_fast)) - amp_slow * (1.0 - math.exp(-H / tau_slow)))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor')