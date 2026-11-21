import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    h = np.array([float(r['H']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (1.0 - np.exp(-h / tau_fast)) - amp_slow * (1.0 - np.exp(-h / tau_slow)))
    result = least_squares(lambda p: pred(p) - y, x0=[1.0, 1.0, 1.0, 10.0], bounds=([0.0, 1e-6, 0.0, 1e-6], ([10.0, 100.0, 10.0, 1000.0])))
    fit.params = result.x
    return fit

def predict(row):
    amp_fast, tau_fast, amp_slow, tau_slow = fit.params
    h = float(row['H'])
    u = float(row['u_now'])
    return float(u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow))))