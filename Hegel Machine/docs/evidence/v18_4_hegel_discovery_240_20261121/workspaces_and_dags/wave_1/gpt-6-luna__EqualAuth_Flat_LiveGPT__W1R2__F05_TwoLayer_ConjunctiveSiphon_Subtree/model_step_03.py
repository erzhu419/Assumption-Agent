import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(1e-8, float(r['u_now'])) for r in rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    # Hydraulic saturating base with gradient inhibition, plus a gated branch.
    def pred(p, u, s):
        vmax, rho, tau, km, amp, threshold, width, sscale = p
        un = u / tau
        base = vmax * (1.0 - rho * s / (1.0 + s)) * un / (km + un)
        z = np.clip((u - threshold) / width, -40.0, 40.0)
        gate = 1.0 / (1.0 + np.exp(-z))
        branch = amp * gate * np.exp(-s / sscale)
        return base + branch
    p0 = [7.5, 0.7, 4.0, 1.2, 1.0, 6.0, 1.5, 1.0]
    lo = [0.01, 0.0, 0.2, 0.05, -5.0, 0.0, 0.1, 0.05]
    hi = [30.0, 0.99, 20.0, 10.0, 8.0, 20.0, 10.0, 10.0]
    res = least_squares(lambda p: pred(p, u, s) - y, p0, bounds=(lo, hi), max_nfev=20000)
    params = res.x
    def predict(row):
        uu = max(1e-8, float(row['u_now']))
        ss = max(0.0, float(row['s_grad']))
        return float(pred(params, np.array([uu]), np.array([ss]))[0])
    return predict

def predict(row):
    # Fallback interface for callers that invoke predict without a fitted closure.
    return float('nan')