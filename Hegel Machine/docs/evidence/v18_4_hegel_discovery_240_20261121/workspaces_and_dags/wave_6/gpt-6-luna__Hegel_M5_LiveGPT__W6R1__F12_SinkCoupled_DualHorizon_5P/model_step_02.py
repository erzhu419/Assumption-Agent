import math
import numpy as np
from scipy.optimize import least_squares


def _predict_params(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    s = float(row['s_grad'])
    u = float(row['u_now'])
    h = float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    base = vmax * sink * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast))
    adapt = amp_adapt * u * (1.0 - math.exp(-h / tau_slow))
    return base - adapt


def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)

    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y

    starts = [
        [6.0, 0.70, 0.30, 0.9, 12.0],
        [5.0, 0.65, 0.50, 0.5, 20.0],
        [8.0, 0.75, 0.20, 1.2, 30.0],
    ]
    lower = [0.01, 0.0, 0.005, 0.0, 0.01]
    upper = [30.0, 0.999, 10.0, 10.0, 200.0]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]


def predict(row):
    p = fit(_MODEL_TRAIN_ROWS)
    return float(_predict_params(row, p))