import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    H = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def predict_params(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (-np.expm1(-H / tau_fast)) - amp_slow * (-np.expm1(-H / tau_slow)))

    def residual(p):
        return predict_params(p) - y

    best = None
    for start in ([0.8, 0.3, 0.12, 12.0], [1.0, 0.5, 0.2, 20.0], [2.0, 0.2, 0.1, 8.0]):
        result = least_squares(residual, start, bounds=([0.0, 1e-4, 0.0, 1e-4], [20.0, 100.0, 20.0, 200.0]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    return tuple(float(v) for v in best.x)

def predict(row):
    # Parameters are supplied by the fitted model tuple from fit(train_rows).
    amp_fast, tau_fast, amp_slow, tau_slow = row['_fit_params']
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (-math.expm1(-h / tau_fast)) - amp_slow * (-math.expm1(-h / tau_slow)))
