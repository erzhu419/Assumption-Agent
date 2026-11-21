import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    def residual(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return np.array([
            r['u_now'] * (amp_fast * (-math.expm1(-r['H'] / tau_fast))
                          - amp_slow * (-math.expm1(-r['H'] / tau_slow))) - r['y']
            for r in rows
        ])
    best = None
    for start in ([1.0, 0.3, 1.0, 10.0], [1.0, 1.0, 1.0, 20.0], [2.0, 0.5, 1.0, 5.0]):
        result = least_squares(residual, start, bounds=([0.0, 1e-5, 0.0, 1e-4], [20.0, 100.0, 20.0, 500.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    amp_fast, tau_fast, amp_slow, tau_slow = _params
    h = row['H']
    return float(row['u_now'] * (amp_fast * (-math.expm1(-h / tau_fast))
                                 - amp_slow * (-math.expm1(-h / tau_slow))))

_params = None
_original_fit = fit
def fit(train_rows):
    global _params
    _params = _original_fit(train_rows)
    return _params
