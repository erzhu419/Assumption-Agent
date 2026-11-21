import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        af, ass, tf, ts = p
        return u * (af * (1.0 - np.exp(-H / tf)) - ass * (1.0 - np.exp(-H / ts)))
    # Multiple scales help initialize the fast and slow time constants.
    best = None
    for tf0, ts0 in [(0.1, 10.0), (0.3, 20.0), (1.0, 8.0), (1.0, 30.0)]:
        p0 = [1.0, 0.5, tf0, ts0]
        res = least_squares(lambda p: pred(p) - y, p0,
                            bounds=([0.0, 0.0, 1e-5, 1e-4], [20.0, 20.0, 10.0, 1e4]),
                            max_nfev=20000)
        err = float(np.sum(res.fun ** 2))
        if best is None or err < best[0]:
            best = (err, res.x)
    return best[1]

def predict(row):
    af, ass, tf, ts = fit(_TRAIN_ROWS)
    u = float(row['u_now'])
    H = float(row['H'])
    return u * (af * (1.0 - math.exp(-H / tf)) - ass * (1.0 - math.exp(-H / ts)))