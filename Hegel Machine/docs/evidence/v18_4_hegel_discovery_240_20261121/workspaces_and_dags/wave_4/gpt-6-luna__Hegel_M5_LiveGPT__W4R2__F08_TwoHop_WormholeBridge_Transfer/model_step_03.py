import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s


def fit(train_rows):
    rows = list(train_rows)
    if not rows:
        raise ValueError('train_rows must not be empty')
    y = np.asarray([float(r['y']) for r in rows], dtype=float)

    def predict_params(p, row):
        b, amp, rate, c, d = p
        u, s = _features(row)
        return b + amp * (1.0 - math.exp(-rate * u)) + (s - 1.0) * (c + d * u)

    def residuals(p):
        return np.asarray([predict_params(p, r) for r in rows]) - y

    result = least_squares(residuals, x0=np.array([0.0, 3.5, 0.8, 0.0, 0.0]),
                           bounds=([-20.0, -30.0, 1e-6, -30.0, -30.0],
                                   [20.0, 30.0, 20.0, 30.0, 30.0]),
                           max_nfev=20000)
    return result.x


def predict(row):
    p = fit(_training_rows)
    b, amp, rate, c, d = p
    u, s = _features(row)
    return float(b + amp * (1.0 - math.exp(-rate * u)) + (s - 1.0) * (c + d * u))
