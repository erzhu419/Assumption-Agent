import numpy as np
from scipy.optimize import least_squares
import math


def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def pred(p):
        af, tf, ass, ts, g = p
        fast = af * (1.0 - np.exp(-h / tf))
        slow = ass * (1.0 - np.exp(-h / ts))
        return u * (fast - slow) + g * s

    best = None
    starts = ([1.0, 1.0, 1.0, 10.0, 0.0], [2.0, 2.0, 2.0, 20.0, 0.0], [1.0, 0.5, 1.0, 8.0, 0.5])
    for p0 in starts:
        res = least_squares(lambda p: pred(p) - y, p0,
                            bounds=([0.0, 1e-3, 0.0, 1e-3, -10.0], [20.0, 100.0, 20.0, 200.0, 10.0]),
                            max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x


def predict(row):
    af, tf, ass, ts, g = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return u * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts))) + g * s
