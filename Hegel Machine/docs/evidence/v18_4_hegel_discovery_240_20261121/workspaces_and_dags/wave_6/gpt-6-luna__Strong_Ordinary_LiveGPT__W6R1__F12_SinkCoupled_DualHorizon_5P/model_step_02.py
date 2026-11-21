import math
import numpy as np
from scipy.optimize import least_squares


def _predict_params(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))


def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    p0 = np.array([4.0, 0.7, 1.0, 1.0, 15.0], dtype=float)
    lower = np.array([0.0, 0.0, 0.0, 1e-4, 0.1], dtype=float)
    upper = np.array([100.0, 1.0, 100.0, 100.0, 500.0], dtype=float)
    result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=20000)
    return result.x


def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else None
    if p is None:
        raise ValueError('predict requires fitted parameters; call fit(train_rows) first')
    return float(_predict_params(row, p))