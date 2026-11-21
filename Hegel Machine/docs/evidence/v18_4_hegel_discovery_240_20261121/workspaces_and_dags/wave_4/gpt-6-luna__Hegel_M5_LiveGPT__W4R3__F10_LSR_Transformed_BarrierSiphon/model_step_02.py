import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    tau = 0.02
    u = np.array([max(0.0, float(r['u_now']) / tau) for r in rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    sat = u / (1.2 + u)
    inh = s / (1.0 + s)
    def pred(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * inh) * sat + c_loc * np.sqrt(u)
    def residual(p):
        return pred(p) - y
    starts = [[6.0, 0.7, 1.0], [10.0, 0.5, 0.0], [3.0, 0.8, 2.0]]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=([0.0, 0.05, -10.0], [100.0, 0.95, 10.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return tuple(float(v) for v in best.x)

def predict(row):
    vmax, rho, c_loc = fit(_FIT_ROWS)
    u = max(0.0, float(row['u_now']) / 0.02)
    s = max(0.0, float(row['s_grad']))
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u))