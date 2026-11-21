import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def pred(p):
        amp_fast, amp_slow, tau_fast, tau_slow = p
        return u * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))

    def residual(p):
        return pred(p) - y

    # Positive amplitudes and ordered timescales encode fast activation followed by slow reversal.
    result = least_squares(
        residual,
        x0=np.array([1.2, 1.0, 0.25, 12.0]),
        bounds=(np.array([0.0, 0.0, 1e-3, 1.0]), np.array([10.0, 10.0, 5.0, 200.0])),
        max_nfev=50000,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    return result.x

def predict(row):
    amp_fast, amp_slow, tau_fast, tau_slow = fit(_fit_rows)
    u = float(row['u_now'])
    h = float(row['H'])
    return float(u * (amp_fast * (-math.expm1(-h / tau_fast)) - amp_slow * (-math.expm1(-h / tau_slow))))