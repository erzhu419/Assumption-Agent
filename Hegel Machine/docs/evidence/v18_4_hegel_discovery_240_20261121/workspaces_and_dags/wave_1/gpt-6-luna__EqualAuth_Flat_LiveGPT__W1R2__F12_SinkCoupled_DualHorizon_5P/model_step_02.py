import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    af, tf, ass, ts, k = p
    u = float(row['u_now'])
    h = float(row['H'])
    g = float(row['s_grad'])
    fast = af * (-np.expm1(-h / tf))
    slow = ass * (-np.expm1(-h / ts))
    sink = k * g * (-np.expm1(-h))
    return u * (fast - slow) - sink

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_pred(p, r) for r in rows]) - y
    starts = [[1.0, 1.0, 1.0, 20.0, 0.1], [2.0, 0.5, 1.0, 30.0, 0.5], [1.0, 2.0, 2.0, 15.0, -0.2]]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.0, 1e-4, 0.0, 1e-4, -10.0], [20.0, 100.0, 20.0, 200.0, 10.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else None
    if p is None:
        raise ValueError('predict requires fitted parameters in row[\'train_rows\']')
    return float(_pred(p, row))