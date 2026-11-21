import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, uu, ss):
        base, amp, rate, drop, threshold = p
        return base + amp * (1.0 - np.exp(-rate * uu)) - drop * np.maximum(0.0, ss - threshold) ** 2
    best = None
    for p0 in ([1.0, 2.0, 1.0, 1.0, 2.5], [0.0, 3.0, 0.5, 2.0, 2.6], [2.0, 1.0, 2.0, 0.5, 2.4]):
        res = least_squares(lambda p: pred(p, u, s) - y, p0,
                            bounds=([-20.0, -20.0, 1e-5, 0.0, 0.0], [20.0, 20.0, 20.0, 20.0, 10.0]),
                            max_nfev=20000)
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    return best.x

def predict(row):
    p = fit(_FIT_ROWS)
    base, amp, rate, drop, threshold = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(base + amp * (1.0 - np.exp(-rate * u)) - drop * max(0.0, s - threshold) ** 2)
