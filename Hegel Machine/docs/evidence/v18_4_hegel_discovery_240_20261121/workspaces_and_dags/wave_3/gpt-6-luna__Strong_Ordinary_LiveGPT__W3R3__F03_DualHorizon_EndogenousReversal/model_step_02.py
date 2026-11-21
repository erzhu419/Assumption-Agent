import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    H = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def predict_params(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (1.0 - np.exp(-H / tau_fast)) - amp_slow * (1.0 - np.exp(-H / tau_slow)))

    def residual(p):
        return predict_params(p) - y

    result = least_squares(
        residual,
        x0=np.array([1.0, 0.5, 0.5, 10.0]),
        bounds=(np.array([0.0, 1e-4, 0.0, 1e-3]), np.array([100.0, 100.0, 100.0, 1000.0])),
        max_nfev=50000,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    p = result.x
    return {'amp_fast': float(p[0]), 'tau_fast': float(p[1]), 'amp_slow': float(p[2]), 'tau_slow': float(p[3])}

def predict(row):
    p = fit([row])
    H = float(row['H'])
    u = float(row['u_now'])
    return float(u * (p['amp_fast'] * (1.0 - math.exp(-H / p['tau_fast'])) - p['amp_slow'] * (1.0 - math.exp(-H / p['tau_slow']))))