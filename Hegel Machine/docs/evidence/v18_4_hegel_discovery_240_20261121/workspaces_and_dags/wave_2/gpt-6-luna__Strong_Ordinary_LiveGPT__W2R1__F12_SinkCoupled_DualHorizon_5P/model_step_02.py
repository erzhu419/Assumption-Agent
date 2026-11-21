import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    starts = [
        [4.0, 0.5, 0.3, 0.8, 15.0],
        [3.0, 0.7, 0.5, 1.0, 25.0],
        [6.0, 0.3, 0.2, 0.5, 10.0],
        [2.0, 0.8, 1.0, 1.5, 40.0],
    ]
    lo = [0.0, 0.0, 0.01, 0.0, 0.6]
    hi = [20.0, 1.0, 5.0, 10.0, 200.0]
    sols = [least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000) for x0 in starts]
    best = min(sols, key=lambda z: float(np.dot(z.fun, z.fun)))
    return {'vmax': float(best.x[0]), 'rho': float(best.x[1]), 'tau_fast': float(best.x[2]), 'amp_adapt': float(best.x[3]), 'tau_slow': float(best.x[4])}

def predict(row):
    p = (row['vmax'], row['rho'], row['tau_fast'], row['amp_adapt'], row['tau_slow'])
    return float(_predict_params(row, p))