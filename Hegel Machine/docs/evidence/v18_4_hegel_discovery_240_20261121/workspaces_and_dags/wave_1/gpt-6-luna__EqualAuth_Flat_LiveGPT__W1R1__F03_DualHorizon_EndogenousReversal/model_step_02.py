import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    H = np.array([float(r['H']) for r in rows], dtype=float)
    u = np.array([float(r['u_now']) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)

    def predict_params(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (-np.expm1(-H / tau_fast)) - amp_slow * (-np.expm1(-H / tau_slow)))

    def residual(p):
        return predict_params(p) - y

    # Multi-start numerical fitting reduces sensitivity to the exchangeable
    # fast/slow parameterization; select the lowest-residual solution.
    best = None
    for initial in ([1.0, 0.5, 0.5, 10.0], [1.0, 1.0, 0.5, 20.0], [2.0, 0.2, 1.0, 5.0], [0.5, 0.3, 0.5, 15.0]):
        result = least_squares(residual, x0=np.array(initial, dtype=float), bounds=([0.0, 1e-6, 0.0, 1e-6], [10.0, 1000.0, 10.0, 1000.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    amp_fast, tau_fast, amp_slow, tau_slow = fit_params
    H = float(row['H'])
    u = float(row['u_now'])
    return float(u * (amp_fast * (1.0 - math.exp(-H / tau_fast)) - amp_slow * (1.0 - math.exp(-H / tau_slow))))

# Fit parameters are retained between fit and predict calls.
fit_params = np.array([1.0, 0.5, 0.5, 10.0], dtype=float)
_original_fit = fit
def fit(train_rows):
    global fit_params
    fit_params = _original_fit(train_rows)
    return fit_params
