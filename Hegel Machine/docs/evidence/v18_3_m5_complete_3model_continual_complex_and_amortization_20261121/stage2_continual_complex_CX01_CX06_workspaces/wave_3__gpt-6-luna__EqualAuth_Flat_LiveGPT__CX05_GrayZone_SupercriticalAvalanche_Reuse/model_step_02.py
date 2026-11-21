import numpy as np
from scipy.optimize import least_squares
import math

# y = vmax*(1-rho*s/(1+s))*u/(1.2+u) + c_loc*sqrt(u)
#     - k_aval*max(0,s-s_crit)*u; tau_scale = 1.
def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
            + c_loc * math.sqrt(max(0.0, un))
            - k_aval * max(0.0, s - s_crit) * un)

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [
        [4.0, 0.5, 0.5, 2.0, 1.4],
        [5.0, 0.3, 0.3, 3.0, 1.45],
        [3.0, 0.7, 0.8, 1.0, 1.35],
    ]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0,
            bounds=([0.0, 0.0, -10.0, 0.0, 1.0], [20.0, 1.0, 10.0, 20.0, 2.0]),
            max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    return float(_model(row, _FIT_PARAMS))