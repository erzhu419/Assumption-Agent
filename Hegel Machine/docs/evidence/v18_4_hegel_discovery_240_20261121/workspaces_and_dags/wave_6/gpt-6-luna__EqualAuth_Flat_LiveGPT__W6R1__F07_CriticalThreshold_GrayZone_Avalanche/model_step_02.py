import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Fit the five parameters of the thresholded sink-gradient avalanche law.
    # Features constant in the supplied data are handled by regularized priors
    # only through the initialization; all target parameters are fitted.
    def pred(p, r):
        vmax, rho, c_loc, k_aval, s_crit = p
        u = max(0.0, float(r['u_now']))
        s = max(0.0, float(r['s_grad']))
        return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
                + c_loc * math.sqrt(u)
                - k_aval * max(0.0, s - s_crit) * u)
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([pred(p, r) for r in train_rows]) - y
    starts = [
        [4.0, 0.3, 0.4, 1.0, 2.6],
        [3.0, 0.2, 0.6, 1.5, 2.55],
        [5.0, 0.5, 0.2, 0.8, 2.7],
    ]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0,
            bounds=([0, 0, 0, 0, 1.5], [20, 2, 5, 10, 4]),
            max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x.tolist()

def predict(row):
    p = fit(_train_rows)
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
                 + c_loc * math.sqrt(u)
                 - k_aval * max(0.0, s - s_crit) * u)
