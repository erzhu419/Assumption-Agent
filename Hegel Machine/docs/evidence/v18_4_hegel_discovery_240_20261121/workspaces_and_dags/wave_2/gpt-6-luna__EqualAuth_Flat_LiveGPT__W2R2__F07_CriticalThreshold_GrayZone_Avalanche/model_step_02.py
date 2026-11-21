import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(1e-8, float(r['u_now'])) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def predict_params(p, uu, ss):
        vmax, rho, c_loc, k_aval, s_crit = p
        return (vmax * (1.0 - rho * ss / (1.0 + ss)) * (uu / (1.2 + uu))
                + c_loc * np.sqrt(uu)
                - k_aval * np.maximum(0.0, ss - s_crit) * uu)

    best = None
    for crit0 in (2.50, 2.60, 2.68):
        p0 = np.array([5.0, 0.5, 0.2, 1.0, crit0])
        res = least_squares(
            lambda p: predict_params(p, u, s) - y,
            p0,
            bounds=([0.0, 0.0, -10.0, 0.0, 2.0], [30.0, 1.0, 10.0, 10.0, 3.2]),
            max_nfev=20000
        )
        score = float(np.sum(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x)
    params = best[1]
    return tuple(float(v) for v in params)

def predict(row):
    vmax, rho, c_loc, k_aval, s_crit = fit_globals
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
                 + c_loc * math.sqrt(u)
                 - k_aval * max(0.0, s - s_crit) * u)

fit_globals = (0.0, 0.0, 0.0, 0.0, 2.6)
_original_fit = fit
def fit(train_rows):
    global fit_globals
    fit_globals = _original_fit(train_rows)
    return fit_globals
