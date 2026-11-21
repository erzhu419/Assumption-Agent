import numpy as np
from scipy.optimize import least_squares

def _basis(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * np.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_basis(r, p) for r in rows]) - y
    starts = [
        np.array([6.5, 0.70, 0.1, 1.0, 2.58]),
        np.array([5.0, 0.72, 0.5, 1.5, 2.55]),
        np.array([8.0, 0.68, -0.3, 0.7, 2.65]),
    ]
    lo = np.array([0.01, 0.0, -5.0, 0.0, 2.50])
    hi = np.array([30.0, 0.99, 5.0, 10.0, 2.68])
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    return best[1]

def predict(row):
    p = fit([row])
    return float(_basis(row, p))