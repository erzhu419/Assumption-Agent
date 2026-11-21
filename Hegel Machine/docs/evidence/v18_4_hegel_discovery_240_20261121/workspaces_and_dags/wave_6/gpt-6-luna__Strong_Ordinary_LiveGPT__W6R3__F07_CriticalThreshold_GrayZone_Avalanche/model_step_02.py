import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(np.maximum(u, 0.0)) - k_aval * np.maximum(0.0, s - s_crit) * u
    best = None
    # Multistart handles the threshold parameter's piecewise-smooth effect.
    for threshold in (2.3, 2.5, 2.65, 2.8, 3.0):
        p0 = np.array([4.0, 0.4, 0.1, 1.0, threshold])
        res = least_squares(lambda p: pred(p) - y, p0,
            bounds=([0.0, 0.0, -10.0, 0.0, 2.0], [100.0, 0.99, 10.0, 10.0, 4.0]),
            max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    u = float(row['u_now']); s = float(row['s_grad'])
    vmax, rho, c_loc, k_aval, s_crit = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u)
