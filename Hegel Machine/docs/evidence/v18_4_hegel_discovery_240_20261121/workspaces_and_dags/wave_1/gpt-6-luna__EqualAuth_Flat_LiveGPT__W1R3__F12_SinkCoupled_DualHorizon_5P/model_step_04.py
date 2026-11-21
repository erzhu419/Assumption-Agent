import math
import numpy as np
from scipy.optimize import least_squares

def _model(row, p):
    af, a_slow, tau_fast, tau_slow, beta = p
    h = float(row['H'])
    u = float(row['u_now'])
    g = float(row['s_grad'])
    return u * (af * (1.0 - math.exp(-h / tau_fast)) - a_slow * (1.0 - math.exp(-h / tau_slow))) + beta * g

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    p0 = np.array([1.0, 1.0, 0.2, 12.0, 0.0], dtype=float)
    lower = np.array([0.0, 0.0, 1e-3, 0.05, -10.0])
    upper = np.array([20.0, 20.0, 10.0, 1000.0, 10.0])
    result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=20000)
    return result.x

def predict(row):
    return float(_model(row, _FIT_PARAMS))