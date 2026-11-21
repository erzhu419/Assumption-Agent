import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    a, b, c, d, k, q = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return a + b*u/(c+u) + d*np.exp(-k*s)*(1.0 + q*u)

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    starts = [np.array([0.0, 4.0, 1.0, 2.0, 1.0, 0.0]), np.array([0.0, 3.0, 0.7, 2.0, 0.5, 0.1])]
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.array([_pred(p, r) for r in train_rows]) - y, x0, max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_pred(p, row))