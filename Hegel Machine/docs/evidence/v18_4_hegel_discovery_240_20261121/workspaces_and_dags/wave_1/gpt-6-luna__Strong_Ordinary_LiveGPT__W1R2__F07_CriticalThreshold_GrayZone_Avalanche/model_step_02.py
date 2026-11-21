import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred_params(p, row):
        vmax, rho, leak = p
        u = float(row['u_now'])
        s = float(row['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) - leak * u
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([pred_params(p, r) for r in rows]) - y
    starts = ([4, 1, 0.1], [8, 1.5, 0.4], [10, 2, 1])
    fits = [least_squares(residual, x0, max_nfev=20000) for x0 in starts]
    p = min(fits, key=lambda z: np.sum(z.fun**2)).x
    return p

def predict(row):
    p = fit(_training_rows)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    vmax, rho, leak = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) - leak * u)
