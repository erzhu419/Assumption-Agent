import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u) - k_aval * np.maximum(0.0, s - s_crit) * u
    lo = [0.1, 0.0, -3.0, 0.0, 2.36]
    hi = [20.0, 0.99, 3.0, 10.0, 3.2]
    starts = [[6.5, 0.70, 0.1, 1.0, 2.6], [6.5, 0.70, 0.1, 2.0, 2.5], [6.5, 0.70, 0.1, 0.5, 2.8]]
    best = None
    for p0 in starts:
        res = least_squares(lambda p: pred(p) - y, p0, bounds=(lo, hi), max_nfev=20000)
        score = float(np.sum(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x)
    return {'params': best[1].tolist()}

def predict(row):
    vmax, rho, c_loc, k_aval, s_crit = fit_params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u) - k_aval * max(0.0, s - s_crit) * u)
