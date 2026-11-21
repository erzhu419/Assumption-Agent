import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    H = float(row['H'])
    return float(row['u_now']) * (amp_fast * (1.0 - math.exp(-H / tau_fast)) - amp_slow * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    # Positive amplitudes and separated timescales encode the short-gain/long-reversal mechanism.
    starts = [
        [1.0, 0.4, 0.8, 5.0],
        [1.5, 0.8, 1.0, 10.0],
        [2.0, 0.2, 1.0, 3.0],
    ]
    sols = []
    for p0 in starts:
        sol = least_squares(residual, p0, bounds=([0.0, 0.02, 0.0, 0.5], [20.0, 5.0, 20.0, 100.0]), max_nfev=20000)
        sols.append(sol)
    best = min(sols, key=lambda s: float(np.dot(s.fun, s.fun)))
    return best.x

def predict(row):
    p = fit(train_rows)
    return _value(row, p)