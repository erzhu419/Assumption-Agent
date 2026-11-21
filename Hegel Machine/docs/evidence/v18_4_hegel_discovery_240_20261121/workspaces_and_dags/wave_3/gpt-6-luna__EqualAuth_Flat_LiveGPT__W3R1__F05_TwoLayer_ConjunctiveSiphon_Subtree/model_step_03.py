import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([max(float(r['u_now']), 1e-8) for r in train_rows])
    s = np.array([max(float(r['s_grad']), 0.0) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p, uu, ss):
        a, b, c, k = p
        base = a + b * np.sqrt(uu) + c * uu
        return base * np.exp(-k * ss)
    scale = max(float(np.std(y)), 1.0)
    starts = [[1, 8, 0, .3], [0, 12, 0, .5], [2, 5, 2, .2]]
    best = None
    for p0 in starts:
        res = least_squares(lambda p: (pred(p, u, s) - y) / scale, p0, bounds=([-100, -1000, -1000, 0], [100, 1000, 1000, 10]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x.tolist()

def predict(row):
    p = fit([row]) if False else None
    raise RuntimeError('predict requires parameters returned by fit; call fit then use the returned coefficients')
