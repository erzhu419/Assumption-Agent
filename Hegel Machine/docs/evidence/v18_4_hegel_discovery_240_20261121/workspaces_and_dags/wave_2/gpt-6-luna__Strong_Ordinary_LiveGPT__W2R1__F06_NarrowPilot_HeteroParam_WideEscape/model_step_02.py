import math
import numpy as np
from scipy.optimize import least_squares


def _predict_params(row, p, branch_sign):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + branch_sign * c_loc * math.sqrt(u)


def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    best = None
    for sign in (1.0, -1.0):
        def residual(p):
            return np.asarray([_predict_params(r, p, sign) for r in rows]) - y
        result = least_squares(residual, x0=np.array([6.0, 0.7, 0.2]),
                               bounds=([0.0, 0.0, 0.0], [30.0, 3.0, 10.0]),
                               max_nfev=20000)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, sign, result.x)
    return {'params': best[2], 'branch_sign': best[1]}


def predict(row):
    # fit(train_rows) returns the parameter dictionary; attach/use it as model
    # state when predicting in the confirmation harness.
    p = _MODEL['params']
    return float(_predict_params(row, p, _MODEL['branch_sign']))