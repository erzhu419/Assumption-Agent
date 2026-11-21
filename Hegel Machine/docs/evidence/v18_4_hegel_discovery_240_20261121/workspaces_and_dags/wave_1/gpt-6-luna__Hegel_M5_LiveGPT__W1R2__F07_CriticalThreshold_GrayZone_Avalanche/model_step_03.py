import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u)
        return base - k_aval * np.maximum(0.0, s - s_crit) * u
    lo = np.array([0.01, 0.05, -2.0, 0.0, 0.0])
    hi = np.array([30.0, 0.95, 2.0, 20.0, 8.0])
    starts = [np.array([6.5, 0.70, 0.1, 1.0, 2.8]), np.array([6.5, 0.70, 0.1, 3.0, 3.0]), np.array([8.0, 0.7, -0.2, 1.0, 2.5])]
    best = None
    for p0 in starts:
        res = least_squares(lambda p: pred(p) - y, p0, bounds=(lo, hi), max_nfev=30000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    params = best.x
    def predict(row):
        vmax, rho, c_loc, k_aval, s_crit = params
        u0 = float(row['u_now'])
        s0 = float(row['s_grad'])
        return float(vmax * (1.0 - rho * s0 / (1.0 + s0)) * (u0 / (1.2 + u0)) + c_loc * np.sqrt(u0) - k_aval * max(0.0, s0 - s_crit) * u0)
    return predict
