import math
import numpy as np
from scipy.optimize import least_squares


def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)


def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    starts = [
        [5.0, 0.7, 0.2, 1.0, 2.55],
        [4.0, 0.5, 0.5, 1.5, 2.65],
        [6.0, 0.9, 0.0, 0.8, 2.5],
    ]
    candidates = [least_squares(residual, x0, bounds=([0.0, 0.0, -5.0, 0.0, 2.0], [30.0, 2.0, 5.0, 10.0, 3.2]), max_nfev=20000) for x0 in starts]
    return min(candidates, key=lambda z: float(np.dot(z.fun, z.fun))).x


def predict(row):
    return float(_model(row, fit(_TRAIN_ROWS)))