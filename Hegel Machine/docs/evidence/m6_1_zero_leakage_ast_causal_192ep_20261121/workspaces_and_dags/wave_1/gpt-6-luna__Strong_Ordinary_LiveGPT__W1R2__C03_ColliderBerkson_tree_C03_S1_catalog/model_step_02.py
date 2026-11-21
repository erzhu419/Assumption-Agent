import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r['u_1']), float(r['u_2'])] for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)

    def predict_params(p, x):
        b0, b1, b2, k = p
        u1, u2 = x[:, 0], x[:, 1]
        return b0 + b1 * np.log1p(u1) + b2 * u2 / (k + u2)

    result = least_squares(
        lambda p: predict_params(p, X) - y,
        x0=np.array([0.0, 2.0, 4.0, 1.5]),
        bounds=([-20.0, -20.0, -20.0, 0.05], [20.0, 20.0, 20.0, 20.0]),
        max_nfev=20000,
    )
    fit.params_ = result.x
    return fit


def predict(row):
    b0, b1, b2, k = fit.params_
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return float(b0 + b1 * math.log1p(u1) + b2 * u2 / (k + u2))