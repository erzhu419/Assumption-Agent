import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(q):
        p = np.exp(q)
        return np.array([_pred(r, p) for r in rows]) - y
    # Positive parameters; initialization covers fast buildup and slower reversal.
    result = least_squares(residual, np.log([1.5, 0.5, 1.0, 10.0]), max_nfev=50000)
    return np.exp(result.x)

def predict(row):
    return float(_pred(row, fit(_TRAIN_ROWS)))