import math
import numpy as np
from scipy.optimize import least_squares

def _f(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    return vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u))*(-math.expm1(-H/tau_fast)) - amp_adapt*u*(-math.expm1(-H/tau_slow))

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[r['u_now'], r['s_grad'], r['H']] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    starts = [np.array([3.0, 0.3, 0.4, 0.2, 8.0]), np.array([4.0, 0.5, 1.0, 0.1, 15.0]), np.array([2.0, 0.2, 0.2, 0.3, 5.0])]
    lo = np.array([0.0, 0.0, 0.01, 0.0, 0.1])
    hi = np.array([20.0, 1.0, 10.0, 5.0, 100.0])
    best = None
    for p0 in starts:
        res = least_squares(lambda p: np.array([_f(p, r) for r in rows])-y, p0, bounds=(lo, hi), max_nfev=30000)
        score = float(np.sum(res.fun**2))
        if best is None or score < best[0]:
            best = (score, res.x)
    return best[1].tolist()

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_f(p, row))