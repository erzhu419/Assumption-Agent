import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    fast_amp, slow_amp, fast_tau, slow_tau = p
    h = float(row['H'])
    return float(row['u_now']) * (fast_amp * (-np.expm1(-h / fast_tau)) - slow_amp * (-np.expm1(-h / slow_tau)))

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(row, p) - float(row['y']) for row in train_rows])
    bounds = ([0.0, 0.0, 0.01, 0.1], [100.0, 100.0, 20.0, 1000.0])
    best = None
    for start in ([2.0, 3.0, 0.8, 10.0], [3.0, 4.0, 1.0, 20.0], [1.5, 2.5, 0.5, 5.0]):
        result = least_squares(residual, start, bounds=bounds, max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _model(row, _params)
