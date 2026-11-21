import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        vmax, rho, amp, threshold, power = p
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        branch = -amp * np.maximum(s - threshold, 0.0) ** power
        return base + branch
    result = least_squares(lambda p: pred(p) - y,
                           x0=np.array([5.0, 0.7, 2.0, 1.2, 1.5]),
                           bounds=([0.01, 0.0, 0.0, 0.0, 0.25],
                                   [30.0, 0.99, 100.0, 3.0, 5.0]),
                           max_nfev=20000)
    return result.x

def predict(row):
    vmax, rho, amp, threshold, power = fit(_FIT_ROWS)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) - amp * max(s - threshold, 0.0) ** power)
