import math
import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    vmax, rho, c_loc, k_aval, s_crit, km = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (km + u))
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_pred(p, r) for r in train_rows]) - y
    starts = [np.array([4.0, 0.5, 15.0, 10.0, 2.5, 0.01]),
              np.array([3.0, 0.3, 20.0, 5.0, 2.6, 0.02]),
              np.array([5.0, 0.7, 10.0, 20.0, 2.4, 0.005])]
    lo = [0.0, 0.0, 0.0, 0.0, 0.0, 1e-6]
    hi = [100.0, 1.0, 200.0, 500.0, 10.0, 1.0]
    results = [least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000) for x0 in starts]
    return min(results, key=lambda result: float(np.dot(result.fun, result.fun))).x

def predict(row):
    params = fit(train_rows)
    return float(_pred(params, row))