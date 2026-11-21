import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    H = np.asarray([float(r['H']) for r in rows])
    u = np.asarray([float(r['u_now']) for r in rows])
    y = np.asarray([float(r['y']) for r in rows])

    def predict_values(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (1.0 - np.exp(-H / tau_fast))
                    - amp_slow * (1.0 - np.exp(-H / tau_slow)))

    best = None
    # Multi-start helps distinguish the fast and slow exponential timescales.
    for start in ([1.0, 0.5, 0.5, 10.0], [1.0, 2.0, 0.5, 30.0], [2.0, 0.2, 1.0, 5.0]):
        result = least_squares(
            lambda p: predict_values(p) - y,
            x0=np.asarray(start, dtype=float),
            bounds=([0.0, 1e-4, 0.0, 1e-4], [100.0, 1e3, 100.0, 1e3]),
            max_nfev=30000
        )
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    return best.x

def predict(row):
    amp_fast, tau_fast, amp_slow, tau_slow = fit(_FIT_ROWS)
    u = float(row['u_now'])
    H = float(row['H'])
    return float(u * (amp_fast * (1.0 - math.exp(-H / tau_fast))
                    - amp_slow * (1.0 - math.exp(-H / tau_slow))))