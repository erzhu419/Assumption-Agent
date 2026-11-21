import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    H = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    # Parameterize tau_slow = tau_fast + delta to enforce positive ordered timescales.
    def predict_params(p, uu, hh):
        amp_fast, amp_slow, tau_fast, delta = p
        tau_slow = tau_fast + delta
        return uu * (amp_fast * (-np.expm1(-hh / tau_fast)) -
                     amp_slow * (-np.expm1(-hh / tau_slow)))

    starts = [
        [1.2, 1.2, 0.5, 12.0],
        [1.0, 1.3, 1.0, 20.0],
        [1.5, 1.5, 0.3, 8.0],
    ]
    best = None
    for p0 in starts:
        res = least_squares(
            lambda p: predict_params(p, u, H) - y,
            x0=np.array(p0, dtype=float),
            bounds=([0.0, 0.0, 1e-4, 1e-4], [10.0, 10.0, 100.0, 200.0]),
            max_nfev=20000
        )
        score = float(np.mean(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x)
    p = best[1]
    return {'amp_fast': float(p[0]), 'amp_slow': float(p[1]),
            'tau_fast': float(p[2]), 'tau_slow': float(p[2] + p[3])}

def predict(row):
    p = fit(_TRAIN_ROWS)
    H = float(row['H'])
    u = float(row['u_now'])
    return u * (p['amp_fast'] * (-math.expm1(-H / p['tau_fast'])) -
                p['amp_slow'] * (-math.expm1(-H / p['tau_slow'])))