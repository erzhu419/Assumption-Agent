import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    basin = (u / (1.2 + u))
    sink = s / (1.0 + s)
    return basin, sink, float(row['x_now']), float(row['mem_conv']), float(row['H'])


def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    F = np.array([_features(r) for r in rows], dtype=float)
    # vmax, rho, x-coupling, memory-coupling, external-leak rate
    def residual(p):
        vmax, rho, bx, bm, leak = p
        pred = vmax * F[:, 0] * (1.0 - rho * F[:, 1]) + bx * F[:, 2] + bm * F[:, 3] - leak * F[:, 4]
        return pred - y
    p0 = np.array([2.5, 0.7, 0.0, 0.0, 0.0])
    res = least_squares(residual, p0, bounds=([0.01, 0.0, -10.0, -10.0, -10.0], [20.0, 1.5, 10.0, 10.0, 10.0]))
    return res.x


def predict(row):
    p = fit(_TRAIN_ROWS)
    basin, sink, x, mem, H = _features(row)
    vmax, rho, bx, bm, leak = p
    return float(vmax * basin * (1.0 - rho * sink) + bx * x + bm * mem - leak * H)
