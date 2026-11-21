import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            * (-np.expm1(-h / tau_fast))
            - amp_adapt * u * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_model(p, r) - float(r['y']) for r in rows])
    best = None
    for start in ([8.0, 0.5, 0.5, 1.0, 10.0],
                  [10.0, 0.8, 1.0, 2.0, 20.0],
                  [5.0, 0.3, 0.2, 1.0, 5.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 1e-4, 0.0, 1e-4], [1000.0, 2.0, 1000.0, 1000.0, 1000.0]), max_nfev=10000)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_model(_params, row))