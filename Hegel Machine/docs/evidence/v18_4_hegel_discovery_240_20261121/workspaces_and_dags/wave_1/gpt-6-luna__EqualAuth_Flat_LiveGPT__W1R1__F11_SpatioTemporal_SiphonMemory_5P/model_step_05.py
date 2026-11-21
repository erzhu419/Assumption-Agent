import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows])
    # Fit both documented local square-root branches and retain the lower-error one.
    def pred(r, p, sign):
        vmax, rho, c_loc = p
        u = max(1e-8, float(r['u_now']))
        s = float(r['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + sign * c_loc * math.sqrt(u)
    best = None
    for sign in (1.0, -1.0):
        def residual(p):
            return np.array([pred(r, p, sign) - float(r['y']) for r in rows])
        res = least_squares(residual, x0=np.array([3.0, 0.5, 0.2]), bounds=([0.0, 0.0, 0.0], [20.0, 5.0, 10.0]), max_nfev=20000)
        score = float(np.mean(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x, sign)
    _, params, sign = best
    def predict(row):
        return float(pred(row, params, sign))
    return predict


def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor.')
