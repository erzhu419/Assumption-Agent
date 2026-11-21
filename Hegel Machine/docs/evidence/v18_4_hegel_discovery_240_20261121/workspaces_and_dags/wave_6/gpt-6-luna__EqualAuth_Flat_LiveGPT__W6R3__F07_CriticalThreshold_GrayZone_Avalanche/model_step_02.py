import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    lower = [0.0, 0.0, 0.0, 0.0, 2.50]
    upper = [20.0, 2.0, 10.0, 10.0, 2.68]
    starts = [[4.0, 0.3, 0.5, 1.0, 2.60], [6.0, 0.5, 1.0, 2.0, 2.58], [3.0, 0.2, 0.2, 3.0, 2.64]]
    fits = [least_squares(residual, x0, bounds=(lower, upper), max_nfev=20000) for x0 in starts]
    p = min(fits, key=lambda z: float(np.sum(z.fun ** 2))).x
    return p.tolist()

def predict(row):
    return float(_predict_params(row, fit_params))