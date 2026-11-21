import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    # Parameters: vmax, rho, c_loc, k_aval, s_crit.
    def pred(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u)) + c_loc*np.sqrt(u) - k_aval*np.maximum(0.0, s-s_crit)*u
    result = least_squares(lambda p: pred(p)-y,
        x0=np.array([5.0, 0.35, 0.0, 1.0, 2.6]),
        bounds=(np.array([0.01, 0.0, -10.0, 0.0, 2.0]),
                np.array([20.0, 1.0, 10.0, 10.0, 3.2])),
        max_nfev=20000)
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u)) + c_loc*math.sqrt(u) - k_aval*max(0.0, s-s_crit)*u)
