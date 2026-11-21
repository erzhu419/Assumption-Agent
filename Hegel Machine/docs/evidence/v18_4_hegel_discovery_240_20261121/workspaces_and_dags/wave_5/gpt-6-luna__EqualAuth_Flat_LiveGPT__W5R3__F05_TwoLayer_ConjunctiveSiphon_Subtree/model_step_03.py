import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # The known two-layer siphon structure: a saturating base field, a
    # conjunctive local/gradient contribution, and a supercritical sink term.
    def pred(p, r):
        vmax, rho, c_loc, k_aval, s_crit = p
        u = max(0.0, float(r['u_now']))
        s = max(0.0, float(r['s_grad']))
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        local = c_loc * math.sqrt(u)
        siphon = k_aval * max(0.0, s - s_crit) * u
        return base + local - siphon
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in train_rows])
    starts = [
        [5.0, 0.3, 0.2, 0.04, 2.6],
        [4.0, 0.5, 0.1, 0.08, 2.5],
        [6.0, 0.2, 0.3, 0.02, 2.7],
    ]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0,0,0,0,2.36],[100,1,10,10,5]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x.tolist()

def predict(row):
    p = fit(row.get('train_rows', []))
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u)