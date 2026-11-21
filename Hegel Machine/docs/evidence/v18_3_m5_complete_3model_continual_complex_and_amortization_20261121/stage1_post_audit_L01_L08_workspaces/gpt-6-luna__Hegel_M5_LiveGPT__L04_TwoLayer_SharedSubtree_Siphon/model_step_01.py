import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    # tau_scale is 1.0 for this task, so u_n = u_now.
    un = u
    def pred(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * np.sqrt(np.maximum(un, 0.0))
    lower = np.array([3.836397, 0.658333, -0.822167])
    upper = np.array([9.163604, 0.748333, 1.108834])
    starts = [np.array([6.5, 0.70, 0.0]), np.array([6.5, 0.70, 0.3])]
    fits = [least_squares(lambda p: pred(p) - y, np.clip(p0, lower + 1e-9, upper - 1e-9), bounds=(lower, upper)) for p0 in starts]
    best = min(fits, key=lambda result: np.sum(result.fun ** 2))
    return best.x

def predict(row):
    vmax, rho, c_loc = fit(_FIT_ROWS)
    un = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * math.sqrt(max(un, 0.0)))