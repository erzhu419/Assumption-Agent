import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def model(p):
        amp_fast, amp_slow, tau_fast, tau_slow = p
        return u * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))

    best = None
    # Positive amplitudes and timescales; multiple starts address nonlinear tau ambiguity.
    for start in ([1.0, 0.5, 1.0, 12.0], [1.5, 0.8, 0.5, 20.0], [2.0, 1.0, 2.0, 30.0], [1.0, 0.3, 0.2, 8.0]):
        res = least_squares(lambda p: model(p) - y, x0=np.array(start), bounds=([0, 0, 1e-4, 1e-4], [np.inf, np.inf, np.inf, np.inf]), max_nfev=20000)
        score = float(np.sum(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x)
    a, b, tf, ts = best[1]
    return {'amp_fast': float(a), 'amp_slow': float(b), 'tau_fast': float(tf), 'tau_slow': float(ts)}

def predict(row):
    p = fit_parameters
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (p['amp_fast'] * (1.0 - math.exp(-h / p['tau_fast'])) - p['amp_slow'] * (1.0 - math.exp(-h / p['tau_slow'])))