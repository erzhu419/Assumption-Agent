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
    lower = np.array([0.05, 0.0, 1e-4, 0.0, 1e-3])
    upper = np.array([30.0, 0.999, 20.0, 10.0, 200.0])
    def residual(p):
        return np.array([_predict_params(r, p) - float(r['y']) for r in train_rows])
    starts = [
        [6.0, 0.7, 0.3, 0.5, 10.0],
        [6.0, 0.7, 0.8, 0.5, 20.0],
        [6.0, 0.7, 0.15, 0.5, 5.0],
    ]
    results = [least_squares(residual, np.clip(np.array(s, dtype=float), lower + 1e-8, upper - 1e-8), bounds=(lower, upper), max_nfev=20000) for s in starts]
    return min(results, key=lambda z: float(np.dot(z.fun, z.fun))).x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else None
    if p is None:
        raise ValueError('predict requires fitted parameters in row[\'params\']')
    return float(_predict_params(row, p))