import math
import numpy as np
from scipy.optimize import least_squares


def _predict_params(row, p):
    vmax, rho, branch = p
    u = max(float(row['u_now']), 1e-9)
    s = float(row['s_grad'])
    un = u  # tau_scale is 1.0 for this task
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    return base + branch * math.sqrt(un)


def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r['u_now']), float(r['s_grad'])] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in rows]) - y
    best = None
    for sign in (-1.0, 1.0):
        p0 = np.array([5.0, 0.5, sign * 0.3])
        result = least_squares(residual, p0, bounds=([0.0, 0.0, -10.0], [30.0, 1.0, 10.0]), max_nfev=20000)
        score = float(np.mean(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]


def predict(row):
    p = fit(_training_rows)
    return float(_predict_params(row, p))