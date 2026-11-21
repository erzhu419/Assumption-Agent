import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    H = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    q = s / (1.0 + s)

    def pred(p):
        af, tf, ass, ts, rho = p
        temporal = af * (-np.expm1(-H / tf)) - ass * (-np.expm1(-H / ts))
        return u * temporal * (1.0 - rho * q)

    best = None
    for start in ([1.0, 1.0, 1.0, 8.0, 0.5], [2.0, 1.0, 1.0, 12.0, 0.2], [1.0, 0.5, 0.5, 20.0, 0.7]):
        result = least_squares(lambda p: pred(p) - y, x0=np.array(start, dtype=float),
            bounds=([0.0, 1e-4, 0.0, 1e-4, -5.0], [20.0, 100.0, 20.0, 200.0, 5.0]),
            max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    return best.x

def predict(row):
    p = fit([row]) if False else _fitted_params
    af, tf, ass, ts, rho = p
    H = float(row['H'])
    s = float(row['s_grad'])
    temporal = af * (-math.expm1(-H / tf)) - ass * (-math.expm1(-H / ts))
    return float(row['u_now']) * temporal * (1.0 - rho * s / (1.0 + s))

_fitted_params = None
