import numpy as np
from scipy.optimize import least_squares

def _predict(p, row):
    vmax, rho, c_sqrt = p
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_sqrt * np.sqrt(u)

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict(p, r) for r in train_rows]) - y
    starts = ([3.0, 0.7, 0.5], [4.0, 0.5, 0.0], [2.0, 0.9, 1.0])
    best = None
    for x0 in starts:
        result = least_squares(residual, x0=x0, bounds=([0.0, 0.0, -10.0], [20.0, 1.0, 10.0]))
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    return best.x

def predict(row):
    return float(_predict(fit(train_rows), row))