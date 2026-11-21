import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow, rho = p
    h = float(row['H'])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    temporal = amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow))
    sink = 1.0 - rho * s / (1.0 + s)
    return u * temporal * sink

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - ys
    best = None
    for start in ([1.0, 2.0, 1.0, 20.0, 0.3], [2.0, 4.0, 1.5, 18.0, 0.6], [1.0, 1.0, 1.0, 30.0, 0.0]):
        result = least_squares(residual, start, bounds=([0.0, 1e-3, 0.0, 1e-3, -2.0], [20.0, 100.0, 20.0, 300.0, 2.0]), max_nfev=30000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return np.asarray(best.x, dtype=float)

def predict(row):
    return float(_pred(row, _FITTED_PARAMS))