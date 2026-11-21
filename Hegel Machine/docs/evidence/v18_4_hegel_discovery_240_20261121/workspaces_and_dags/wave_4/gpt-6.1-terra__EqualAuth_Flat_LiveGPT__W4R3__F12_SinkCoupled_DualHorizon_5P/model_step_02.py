import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(p, r) - float(r['y']) for r in train_rows])
    best = None
    for initial in ([5.0, 0.5, 0.3, 1.0, 10.0], [10.0, 1.0, 1.0, 2.0, 20.0], [3.0, 0.2, 0.1, 1.0, 5.0]):
        result = least_squares(residual, initial, bounds=([0.0, 0.0, 1e-5, 0.0, 1e-5], [1000.0, 3.0, 100.0, 100.0, 1000.0]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_model(_params, row))