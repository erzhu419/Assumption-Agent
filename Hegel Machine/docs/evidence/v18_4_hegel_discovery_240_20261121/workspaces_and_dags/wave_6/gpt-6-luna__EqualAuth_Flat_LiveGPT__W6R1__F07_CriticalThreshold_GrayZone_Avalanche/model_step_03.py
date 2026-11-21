import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Avalanche law: saturating base-field response with a thresholded gradient sink.
    def pred(p, r):
        vmax, rho, c_loc, k_aval, s_crit = p
        u = max(1e-8, float(r['u_now']))
        s = float(r['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in train_rows])
    starts = [
        [5.0, 0.5, 0.2, 1.0, 2.6],
        [4.0, 0.4, 0.3, 1.2, 2.5],
        [6.0, 0.7, 0.1, 0.8, 2.7]
    ]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.0, -3.0, -5.0, 0.0, 0.0], [50.0, 3.0, 5.0, 20.0, 10.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(_train_rows)
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u)
