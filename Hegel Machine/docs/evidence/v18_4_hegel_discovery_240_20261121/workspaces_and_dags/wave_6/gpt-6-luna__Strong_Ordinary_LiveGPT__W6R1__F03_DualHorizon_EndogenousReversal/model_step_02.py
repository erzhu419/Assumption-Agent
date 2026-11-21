import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def model(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (1.0 - np.exp(-h / tau_fast))
                    - amp_slow * (1.0 - np.exp(-h / tau_slow)))

    best = None
    # Positive amplitudes and ordered times encode the fast gain / slow adaptation mechanism.
    for tf0, ts0 in ((0.2, 5.0), (0.5, 10.0), (1.0, 20.0), (2.0, 40.0)):
        p0 = np.array([1.0, tf0, 1.0, ts0])
        result = least_squares(
            lambda p: model(p) - y,
            p0,
            bounds=([0.0, 1e-5, 0.0, 1e-4], [20.0, 10.0, 20.0, 200.0]),
            max_nfev=20000,
        )
        # Require tau_slow > tau_fast; discard the reversed labeling.
        if result.x[3] <= result.x[1]:
            continue
        score = float(np.sum((model(result.x) - y) ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    if best is None:
        result = least_squares(
            lambda p: model(p) - y,
            [1.0, 0.5, 1.0, 10.0],
            bounds=([0.0, 1e-5, 0.0, 1e-4], [20.0, 10.0, 20.0, 200.0]),
            max_nfev=20000,
        )
        p = result.x
    else:
        p = best[1]
    return {'amp_fast': float(p[0]), 'tau_fast': float(p[1]),
            'amp_slow': float(p[2]), 'tau_slow': float(p[3])}

def predict(row):
    p = fit([row]) if not isinstance(row, dict) or '_fit_params' not in row else row['_fit_params']
    u = float(row['u_now'])
    h = float(row['H'])
    return u * (p['amp_fast'] * (1.0 - math.exp(-h / p['tau_fast']))
                - p['amp_slow'] * (1.0 - math.exp(-h / p['tau_slow'])))