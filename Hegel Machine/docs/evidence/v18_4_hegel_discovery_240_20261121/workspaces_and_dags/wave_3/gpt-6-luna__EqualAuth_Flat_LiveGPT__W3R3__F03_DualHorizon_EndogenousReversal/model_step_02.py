import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def predict_params(p):
        vmax, half, amp, tau_fast, tau_slow = p
        return vmax * (u / (half + u)) * (-np.expm1(-h / tau_fast)) - amp * u * (-np.expm1(-h / tau_slow))
    result = least_squares(lambda p: predict_params(p) - y,
                           x0=[8.0, 1.2, 1.0, 1.0, 10.0],
                           bounds=([0.0, 1e-6, 0.0, 1e-5, 1e-5],
                                   [100.0, 100.0, 100.0, 100.0, 1000.0]),
                           max_nfev=50000)
    return result.x.tolist()

def predict(row):
    vmax, half, amp, tau_fast, tau_slow = fit_params
    u = float(row['u_now'])
    h = float(row['H'])
    return float(vmax * (u / (half + u)) * (-math.expm1(-h / tau_fast)) - amp * u * (-math.expm1(-h / tau_slow)))

fit_params = None

def fit(train_rows):
    global fit_params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def model(p):
        vmax, half, amp, tau_fast, tau_slow = p
        return vmax * (u / (half + u)) * (-np.expm1(-h / tau_fast)) - amp * u * (-np.expm1(-h / tau_slow))
    result = least_squares(lambda p: model(p) - y,
                           x0=[8.0, 1.2, 1.0, 1.0, 10.0],
                           bounds=([0.0, 1e-6, 0.0, 1e-5, 1e-5], [100.0, 100.0, 100.0, 100.0, 1000.0]),
                           max_nfev=50000)
    fit_params = result.x
    return fit_params.tolist()

def predict(row):
    vmax, half, amp, tau_fast, tau_slow = fit_params
    u, h = float(row['u_now']), float(row['H'])
    return float(vmax * (u / (half + u)) * (-math.expm1(-h / tau_fast)) - amp * u * (-math.expm1(-h / tau_slow)))