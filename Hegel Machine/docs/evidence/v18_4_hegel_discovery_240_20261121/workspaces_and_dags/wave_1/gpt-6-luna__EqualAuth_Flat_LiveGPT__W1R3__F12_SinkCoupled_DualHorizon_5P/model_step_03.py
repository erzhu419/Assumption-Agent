import math
import numpy as np
from scipy.optimize import least_squares

def _features(row, p):
    amp_fast, amp_slow, tau_fast, tau_gap, sink = p
    tau_slow = tau_fast + tau_gap
    h = float(row['H'])
    gain = amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow))
    return float(row['u_now']) * gain + sink * float(row['s_grad'])

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_features(r, p) for r in rows]) - y
    p0 = np.array([2.0, 3.0, 0.3, 5.0, 0.0])
    lower = np.array([0.0, 0.0, 1e-4, 1e-4, -10.0])
    upper = np.array([20.0, 20.0, 10.0, 1000.0, 10.0])
    result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=20000)
    return result.x

def predict(row):
    return float(_features(row, fit_rows))