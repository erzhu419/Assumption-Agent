import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    # Saturating hub flux with sink inhibition and an open square-root branch.
    def pred(p, r):
        vmax, rho, c = p
        un = max(1e-9, r['u_now'])
        sg = r['s_grad']
        return vmax * (1.0 - rho * sg / (1.0 + sg)) * (un / (1.2 + un)) + c * math.sqrt(un)
    def residual(p):
        return np.array([pred(p, r) - r['y'] for r in rows])
    best = None
    for x0 in ([3, .7, .2], [6, .5, -.3], [2, .9, 1.0], [8, .8, -1]):
        res = least_squares(residual, x0, bounds=([0, 0, -10], [100, 1.5, 10]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    u = max(1e-9, row['u_now'])
    s = row['s_grad']
    return float(p[0] * (1.0 - p[1] * s / (1.0 + s)) * (u / (1.2 + u)) + p[2] * math.sqrt(u))