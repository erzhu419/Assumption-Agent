import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    H = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def predict_params(p):
        amp_fast, amp_slow = p[0], p[1]
        tau_fast, tau_slow = np.exp(p[2]), np.exp(p[3])
        return u * (amp_fast * (-np.expm1(-H / tau_fast)) - amp_slow * (-np.expm1(-H / tau_slow)))

    def residual(p):
        return predict_params(p) - y

    best = None
    for tf in (0.05, 0.2, 0.8, 2.0):
        for ts in (2.0, 8.0, 25.0, 80.0):
            p0 = np.array([1.5, 1.0, math.log(tf), math.log(ts)])
            result = least_squares(residual, p0, max_nfev=20000)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    return best.x


def predict(row):
    p = fit(_TRAIN_ROWS)
    H = float(row['H'])
    u = float(row['u_now'])
    tau_fast, tau_slow = np.exp(p[2]), np.exp(p[3])
    return float(u * (p[0] * (-math.expm1(-H / tau_fast)) - p[1] * (-math.expm1(-H / tau_slow))))